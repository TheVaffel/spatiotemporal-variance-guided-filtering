 
#include "svgf.hpp"
#include "utils.hpp"

#include "OpenImageIO/imageio.h"

#include <iostream>

#include "CLUtils/CLUtils.hpp"

#include <glm/glm.hpp>

namespace OpenImageIO = OIIO;

int main() {

    std::cout << "Initialize" << std::endl;


    ImageData image_data = initializeData();

	
    clutils::CLEnv clEnv;
    cl::Context &context(clEnv.addContext(PLATFORM_INDEX));

    // Find name of the used device
    std::string deviceName;
    clEnv.devices[0][DEVICE_INDEX].getInfo(CL_DEVICE_NAME, &deviceName);
    printf("Using device named: %s\n", deviceName.c_str());

    cl::CommandQueue &queue(clEnv.addQueue(0, DEVICE_INDEX, CL_QUEUE_PROFILING_ENABLE));


    std::stringstream build_options;
    build_options << " -D IMAGE_WIDTH=" << IMAGE_WIDTH
		  << " -D IMAGE_HEIGHT=" << IMAGE_HEIGHT
		  << " -D NUM_FRAMES=" << NUM_FRAMES;

    
    cl::Kernel reproject_kernel(clEnv.addProgram(0, "svgf.cl", "reproject",
						 build_options.str().c_str()));
    cl::Kernel variance_kernel(clEnv.addProgram(0, "svgf.cl", "compute_variance",
						build_options.str().c_str()));
    cl::Kernel atrous_kernel(clEnv.addProgram(0, "svgf.cl", "atrous",
					      build_options.str().c_str()));

    cl::NDRange global_range(IMAGE_WIDTH, IMAGE_HEIGHT);
    cl::NDRange local_range(LOCAL_SIZE_X, LOCAL_SIZE_Y);

    
    Double_buffer<cl::Buffer> accumulated_buffer(context,
						 CL_MEM_READ_WRITE, IMAGE_SIZE * 3 * sizeof(cl_float));
    Double_buffer<cl::Buffer> normal_buffer(context,
					    CL_MEM_READ_WRITE, IMAGE_SIZE * 3 * sizeof(cl_float));
    Double_buffer<cl::Buffer> position_buffer(context,
					      CL_MEM_READ_WRITE, IMAGE_SIZE * 3 * sizeof(cl_float));

    // Util buffer: x = spp, y = moment, z = linear_z 
    Double_buffer<cl::Buffer> util_buffer(context,
					  CL_MEM_READ_WRITE, IMAGE_SIZE * 3 * sizeof(cl_float));
    Double_buffer<cl::Buffer> temp_atrous_buffer(context,
						 CL_MEM_READ_WRITE, IMAGE_SIZE * 4 * sizeof(cl_float));
    
    Double_buffer<cl::Buffer> in_buffer(context,
					CL_MEM_READ_WRITE, IMAGE_SIZE * 3 * sizeof(cl_float));
    cl::Buffer albedo_buffer(context,
			     CL_MEM_READ_WRITE, IMAGE_SIZE * 3 * sizeof(cl_float));
    
    cl::Buffer output_buffer(context,
			     CL_MEM_READ_WRITE, IMAGE_SIZE * 3 * sizeof(cl_float));


    std::vector<clutils::GPUTimer<std::milli>> reproject_timer;
    reproject_timer.assign(NUM_FRAMES - 1, clutils::GPUTimer<std::milli>(clEnv.devices[0][0]));

    std::vector<clutils::GPUTimer<std::milli>> variance_timer;
    variance_timer.assign(NUM_FRAMES - 1, clutils::GPUTimer<std::milli>(clEnv.devices[0][0]));
    
    std::vector<clutils::GPUTimer<std::milli>> atrous_timer[NUM_ATROUS_ITERATIONS];
    for(int i = 0; i < NUM_ATROUS_ITERATIONS; i++) {
	atrous_timer[i].assign(NUM_FRAMES - 1, clutils::GPUTimer<std::milli>(clEnv.devices[0][0]));
    }

    clutils::ProfilingInfo<NUM_FRAMES - 1> profile_info_reproject("Reprojecting samples");
    clutils::ProfilingInfo<NUM_FRAMES - 1> profile_info_variance("Computing variance");
    clutils::ProfilingInfo<NUM_FRAMES - 1> profile_info_atrous("Running atrous multiple iterations");
    clutils::ProfilingInfo<NUM_FRAMES - 1> profile_info_total("Total");

    for(int frame = 0; frame < NUM_FRAMES; frame++) {
	queue.enqueueWriteBuffer(*in_buffer.current(), true, 0, IMAGE_SIZE * 3 * sizeof(cl_float),
				 image_data.noisy_input[frame].data());
	queue.enqueueWriteBuffer(*normal_buffer.current(), true, 0, IMAGE_SIZE * 3 * sizeof(cl_float),
				 image_data.normals[frame].data());
	queue.enqueueWriteBuffer(*position_buffer.current(), true, 0, IMAGE_SIZE * 3 * sizeof(cl_float),
				 image_data.positions[frame].data());
	queue.enqueueWriteBuffer(albedo_buffer, true, 0, IMAGE_SIZE * 3 * sizeof(cl_float),
				 image_data.albedos[frame].data());
	

	glm::mat4 curr_view_matrix = (*(glm::mat4*)inverse_perspective_matrix) * (*(glm::mat4*)(&camera_matrices[frame][0][0]));
	glm::vec4 cur_mat_row = glm::vec4(curr_view_matrix[0][2],
					  curr_view_matrix[1][2],
					  curr_view_matrix[2][2],
					  curr_view_matrix[3][2]);
	
	int safe_prev = frame == 0 ? 0 : frame - 1;
	int arg_index = 0;
	reproject_kernel.setArg(arg_index++, *in_buffer.current());
	reproject_kernel.setArg(arg_index++, *normal_buffer.current());
	reproject_kernel.setArg(arg_index++, *normal_buffer.previous());
	reproject_kernel.setArg(arg_index++, *position_buffer.current());
	reproject_kernel.setArg(arg_index++, *position_buffer.previous());
	reproject_kernel.setArg(arg_index++, *accumulated_buffer.current());
	reproject_kernel.setArg(arg_index++, *accumulated_buffer.previous());
	reproject_kernel.setArg(arg_index++, *util_buffer.current());
	reproject_kernel.setArg(arg_index++, *util_buffer.previous());
	reproject_kernel.setArg(arg_index++, sizeof(cl_float4), &cur_mat_row[0]);
	reproject_kernel.setArg(arg_index++, sizeof(cl_float16), &camera_matrices[safe_prev][0][0]);
	reproject_kernel.setArg(arg_index++, sizeof(cl_int), &frame);
	

	int res = queue.enqueueNDRangeKernel(reproject_kernel, cl::NullRange, global_range,
					     local_range, nullptr, &reproject_timer[safe_prev].event());
	
	
	arg_index = 0;
	variance_kernel.setArg(arg_index++, *normal_buffer.current());
	variance_kernel.setArg(arg_index++, *accumulated_buffer.current());
	variance_kernel.setArg(arg_index++, *util_buffer.current());
	variance_kernel.setArg(arg_index++, *temp_atrous_buffer.current());

	queue.enqueueNDRangeKernel(variance_kernel, cl::NullRange, global_range,
				   local_range, nullptr, &variance_timer[safe_prev].event());
	

	arg_index = 0;
	atrous_kernel.setArg(arg_index++, *normal_buffer.current());
	atrous_kernel.setArg(arg_index++, albedo_buffer);
	atrous_kernel.setArg(arg_index++, *util_buffer.current());
	
	for(int ai = 0; ai < NUM_ATROUS_ITERATIONS; ai++) {
	    int step_size = 1 << ai;
	    int last = ai == NUM_ATROUS_ITERATIONS - 1;
	    
	    arg_index = 3;
	    atrous_kernel.setArg(arg_index++, *temp_atrous_buffer.current());
	    
	    if(last) {
		atrous_kernel.setArg(arg_index++, output_buffer);
	    } else {
		atrous_kernel.setArg(arg_index++, *temp_atrous_buffer.previous()); // Previous is now output
	    }
	    
	    atrous_kernel.setArg(arg_index++, sizeof(cl_int), &step_size);
	    atrous_kernel.setArg(arg_index++, sizeof(cl_int), &last);

	    queue.enqueueNDRangeKernel(atrous_kernel, cl::NullRange, global_range,
				       local_range, nullptr, &atrous_timer[ai][safe_prev].event());
	    
	    temp_atrous_buffer.swap();
	}

	res = queue.enqueueReadBuffer(output_buffer, false, 0,
				      IMAGE_SIZE * 3 * sizeof(cl_float), image_data.out_data[frame].data());
	
	normal_buffer.swap();
	position_buffer.swap();
	accumulated_buffer.swap();
	util_buffer.swap();
	in_buffer.swap();
        
    }
    
    queue.finish();

    for(int i = 0 ; i < NUM_FRAMES - 1; i++) {

	// profile_info_test[i] = test_timer[i].duration();
	profile_info_reproject[i] = reproject_timer[i].duration();
	profile_info_variance[i] = variance_timer[i].duration();

	cl_ulong atrous_start =
	    atrous_timer[0][i].event().getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong atrous_end =
	    atrous_timer[NUM_ATROUS_ITERATIONS - 1][i].event().getProfilingInfo<CL_PROFILING_COMMAND_END>();
	profile_info_atrous[i] =
	    (atrous_end - atrous_start) * atrous_timer[0][i].getUnit();

	cl_ulong total_start =
	    reproject_timer[i].event().getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong total_end = atrous_end;

	profile_info_total[i] = (total_end - total_start) * atrous_timer[0][i].getUnit();
    }

    // profile_info_test.print(std::cout);
    profile_info_reproject.print(std::cout);
    profile_info_variance.print(std::cout);
    profile_info_atrous.print(std::cout);
    profile_info_total.print(std::cout);

    writeOutputImages(image_data, "output/output");
    
    return 0;
    
}
