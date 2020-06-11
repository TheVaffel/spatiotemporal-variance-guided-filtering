#include "utils.hpp"

#include <OpenImageIO/imageio.h>


ImageData initializeData() {
    ImageData image_data;
    // printf("Loading input data.\n");
    std::cout << "Loading input data" << std::endl;
    bool error = false;
  
#pragma omp parallel for
    for (int frame = 0; frame < NUM_FRAMES; ++frame)
    {
        if (error)
            continue;

        image_data.out_data[frame].resize(3 * IMAGE_SIZE);

        image_data.albedos[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        Operation_result result = load_image(image_data.albedos[frame].data(), ALBEDO_FILE_NAME,
					     frame);
        if (!result.success)
        {
            error = true;
            printf("Albedo buffer loading failed, reason: %s\n",
                   result.error_message.c_str());
            continue;
        }

        image_data.normals[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        result = load_image(image_data.normals[frame].data(), NORMAL_FILE_NAME, frame);
        if (!result.success)
        {
            error = true;
            printf("Normal buffer loading failed, reason: %s\n",
                   result.error_message.c_str());
            continue;
        }

        image_data.positions[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        result = load_image(image_data.positions[frame].data(), POSITION_FILE_NAME, frame);
        if (!result.success)
        {
            error = true;
            printf("Position buffer loading failed, reason: %s\n",
                   result.error_message.c_str());
            continue;
        }

        image_data.noisy_input[frame].resize(3 * IMAGE_WIDTH * IMAGE_HEIGHT);
        result = load_image(image_data.noisy_input[frame].data(), NOISY_FILE_NAME, frame);
        if (!result.success)
        {
            error = true;
            printf("Noisy buffer loading failed, reason: %s\n",
                   result.error_message.c_str());
            continue;
        }

	std::cout << "Read buffers for frame " << frame << std::endl;
    }

    if (error)
    {
        printf("One or more errors occurred during buffer loading\n");
        exit(-1);
    }

    return image_data;
}


  
Operation_result read_image_file(
    const std::string &file_name, const int frame, float *buffer)
{
    std::unique_ptr<OIIO::ImageInput> in = OIIO::ImageInput::open(
        file_name + std::to_string(frame) + ".exr");
    if (!in || in->spec().width != IMAGE_WIDTH ||
        in->spec().height != IMAGE_HEIGHT || in->spec().nchannels != 3)
    {
	if(!in) {
	    std::cerr << "Could not open " << (file_name + std::to_string(frame) + ".exr") << std::endl;
	}
	// std::cout << "in = " << in << std::endl;
	// std::cout << "width = " << in->spec().width << ", height = " << in->spec().height << ", nchannels = " << in->spec().nchannels << std::endl;
        return {false, "Can't open image file or it has wrong type: " + file_name};
    }

    // NOTE: this converts .exr files that might be in halfs to single precision floats
    // In the dataset distributed with the BMFR paper all exr files are in single precision
    in->read_image(OIIO::TypeDesc::FLOAT, buffer);
    in->close();

    return {true};
}

Operation_result load_image(cl_float *image, const std::string file_name, const int frame)
{
    Operation_result result = read_image_file(file_name, frame, image);
    if (!result.success)
        return result;

    return {true};
}

void writeOutputImages(const ImageData& image_data, const std::string& output_file) {
  
    // Store results
    bool error = false;
#pragma omp parallel for
    for (int frame = 0; frame < NUM_FRAMES; ++frame)
    {
        if (error)
            continue;

        // Output image
        std::string output_file_name = output_file + std::to_string(frame) + ".png";
        // Crops back from WORKSET_SIZE to IMAGE_SIZE
        OIIO::ImageSpec spec(IMAGE_WIDTH, IMAGE_HEIGHT, 3,
			     OIIO::TypeDesc::FLOAT);
        std::unique_ptr<OIIO::ImageOutput>
            out(OIIO::ImageOutput::create(output_file_name));
        if (out && out->open(output_file_name, spec))
        {
            out->write_image(OIIO::TypeDesc::FLOAT, image_data.out_data[frame].data(),
                             3 * sizeof(cl_float), IMAGE_WIDTH * 3 * sizeof(cl_float), 0);
            out->close();
        }
        else
        {
            printf("Can't create image file on disk to location %s\n",
                   output_file_name.c_str());
            error = true;
            continue;
        }
    }

    if (error)
    {
        printf("One or more errors occurred during image saving\n");
	exit(-1);
    }
    

    printf("Wrote images with format %s\n", output_file.c_str());
  
}



// Copied from https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes

const char *getErrorString(cl_int error)
{
    switch(error){
	// run-time and JIT compiler errors
    case 0: return "CL_SUCCESS";
    case -1: return "CL_DEVICE_NOT_FOUND";
    case -2: return "CL_DEVICE_NOT_AVAILABLE";
    case -3: return "CL_COMPILER_NOT_AVAILABLE";
    case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
    case -5: return "CL_OUT_OF_RESOURCES";
    case -6: return "CL_OUT_OF_HOST_MEMORY";
    case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
    case -8: return "CL_MEM_COPY_OVERLAP";
    case -9: return "CL_IMAGE_FORMAT_MISMATCH";
    case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
    case -11: return "CL_BUILD_PROGRAM_FAILURE";
    case -12: return "CL_MAP_FAILURE";
    case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
    case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
    case -15: return "CL_COMPILE_PROGRAM_FAILURE";
    case -16: return "CL_LINKER_NOT_AVAILABLE";
    case -17: return "CL_LINK_PROGRAM_FAILURE";
    case -18: return "CL_DEVICE_PARTITION_FAILED";
    case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

	// compile-time errors
    case -30: return "CL_INVALID_VALUE";
    case -31: return "CL_INVALID_DEVICE_TYPE";
    case -32: return "CL_INVALID_PLATFORM";
    case -33: return "CL_INVALID_DEVICE";
    case -34: return "CL_INVALID_CONTEXT";
    case -35: return "CL_INVALID_QUEUE_PROPERTIES";
    case -36: return "CL_INVALID_COMMAND_QUEUE";
    case -37: return "CL_INVALID_HOST_PTR";
    case -38: return "CL_INVALID_MEM_OBJECT";
    case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
    case -40: return "CL_INVALID_IMAGE_SIZE";
    case -41: return "CL_INVALID_SAMPLER";
    case -42: return "CL_INVALID_BINARY";
    case -43: return "CL_INVALID_BUILD_OPTIONS";
    case -44: return "CL_INVALID_PROGRAM";
    case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
    case -46: return "CL_INVALID_KERNEL_NAME";
    case -47: return "CL_INVALID_KERNEL_DEFINITION";
    case -48: return "CL_INVALID_KERNEL";
    case -49: return "CL_INVALID_ARG_INDEX";
    case -50: return "CL_INVALID_ARG_VALUE";
    case -51: return "CL_INVALID_ARG_SIZE";
    case -52: return "CL_INVALID_KERNEL_ARGS";
    case -53: return "CL_INVALID_WORK_DIMENSION";
    case -54: return "CL_INVALID_WORK_GROUP_SIZE";
    case -55: return "CL_INVALID_WORK_ITEM_SIZE";
    case -56: return "CL_INVALID_GLOBAL_OFFSET";
    case -57: return "CL_INVALID_EVENT_WAIT_LIST";
    case -58: return "CL_INVALID_EVENT";
    case -59: return "CL_INVALID_OPERATION";
    case -60: return "CL_INVALID_GL_OBJECT";
    case -61: return "CL_INVALID_BUFFER_SIZE";
    case -62: return "CL_INVALID_MIP_LEVEL";
    case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
    case -64: return "CL_INVALID_PROPERTY";
    case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
    case -66: return "CL_INVALID_COMPILER_OPTIONS";
    case -67: return "CL_INVALID_LINKER_OPTIONS";
    case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

	// extension errors
    case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
    case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
    case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
    case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
    case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
    case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
    default: return "Unknown OpenCL error";
    }
}
