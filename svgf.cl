
static inline float3 linear_to_srgb(float3 x) {
    float3 r;
  
    // Adapted from cuda_utils.h in TwinkleBear's ChameleonRT
    if(x.x <= 0.0031308f) {
	r.x = 12.92f * x.x;
    } else {
	r.x = 1.055f * powr(x.x, 1.f/2.4f) - 0.055f;
    }

    if(x.y <= 0.0031308f) {
	r.y = 12.92f * x.y;
    } else {
	r.y = 1.055f * powr(x.y, 1.f/2.4f) - 0.055f;
    }

    if(x.z <= 0.0031308f) {
	r.z = 12.92f * x.z;
    } else {
	r.z = 1.055f * powr(x.z, 1.f/2.4f) - 0.055f;
    }
  
    return r;
}



#define load_float3(buffer, index) ((float3)				\
	{buffer[(index) * 3], buffer[(index) * 3 + 1], buffer[(index) * 3 + 2]})

#define load_float4(buffer, index) ((float4)				\
	{buffer[(index) * 4], buffer[(index) * 4 + 1], buffer[(index) * 4 + 2], buffer[(index) * 4 + 3]})

static inline void store_float3(
    __global float* restrict buffer,
    const int index,
    const float3 value){

    buffer[index * 3 + 0] = value.x;
    buffer[index * 3 + 1] = value.y;
    buffer[index * 3 + 2] = value.z;
}

static inline void store_float4(
    __global float* restrict buffer,
    const int index,
    const float4 value) {
    buffer[index * 4 + 0] = value.x;
    buffer[index * 4 + 1] = value.y;
    buffer[index * 4 + 2] = value.z;
    buffer[index * 4 + 3] = value.w;
}

static inline float luminance(float3 c) {
    return c.x * 0.2126 + c.y * 0.7152 + c.z * 0.0722;
}

static inline int linear(int2 p) {
    return p.y * IMAGE_WIDTH + p.x;
}

__kernel void test(const __global float* restrict noisy_input,
		   __global float* restrict output) {

    const int2 gid = {get_global_id(0), get_global_id(1)};
    const int linear_pixel = gid.y * IMAGE_WIDTH + gid.x;

    const float3 in_color = load_float3(noisy_input, linear_pixel);

    const float3 vp = (float3){(float)gid.x / (float)IMAGE_WIDTH, 0, (float)gid.y / (float)IMAGE_HEIGHT};

    float3 res = in_color * vp;

    store_float3(output, linear_pixel, res);
}


__kernel void reproject(const __global float* restrict input_noise,
			const __global float* restrict curr_normals,
			const __global float* restrict prev_normals,
			const __global float* restrict curr_positions,
			const __global float* restrict prev_positions,
			__global float* restrict curr_accumulated,
			const __global float* restrict prev_accumulated,
			__global float* restrict curr_util,
			const __global float* restrict prev_util,
			const float4 curr_view_matrix_z_row,
			const float16 prev_cam,
			const int frame_number) {

    
    const float NORMAL_TOLERANCE = 5.0e-2;
    const float POSITION_TOLERANCE = 1e-2;
    
    const int2 gid = {get_global_id(0), get_global_id(1)};
    const int linear_pixel = linear(gid);
    if(gid.x >= IMAGE_WIDTH || gid.y >= IMAGE_HEIGHT) {
	return;
    }

    float3 noise_load = load_float3(input_noise, linear_pixel);
    float3 pos_load = load_float3(curr_positions, linear_pixel);
    float4 world_position = (float4){pos_load.x, pos_load.y, pos_load.z, 1.0};
    
    float z_coord = dot(curr_view_matrix_z_row, world_position);
    float linear_z = fabs(z_coord);

    float3 util_val;
    float3 acc_out = noise_load;
    
    if(frame_number == 0) {
	// store_float3(curr_accumulated, linear_pixel, noise_load);

	float lum = luminance(noise_load);
	util_val.x = 1.0;
	util_val.y = lum * lum;
	util_val.z = linear_z;
	// store_float3(curr_util, linear_pixel, util_val);
	// return;
    } else {
    
	float3 util_load = load_float3(prev_util, linear_pixel);
	float3 normal_load = load_float3(curr_normals, linear_pixel);
    

	float2 position_in_prev_frame = (float2){ dot(prev_cam.s048c, world_position),
						  dot(prev_cam.s159d, world_position)};
	float pos_w = dot(prev_cam.s37bf, world_position);
	position_in_prev_frame /= pos_w;
	position_in_prev_frame += 1.f;
	position_in_prev_frame /= 2.f;

	position_in_prev_frame *= (float2){IMAGE_WIDTH, IMAGE_HEIGHT};
	position_in_prev_frame -= (float2){0.5, 0.5};

	int2 prev_frame_pixel = convert_int2_rtn(position_in_prev_frame);

	int2 offsets[4];
	offsets[0] = (int2){0, 0};
	offsets[1] = (int2){1, 0};
	offsets[2] = (int2){0, 1};
	offsets[3] = (int2){1, 1};

	float2 pix_fract = position_in_prev_frame - convert_float2(prev_frame_pixel);
	float2 inv_pix_fract = 1.0f - pix_fract;
	float weights[4];
	weights[0] = inv_pix_fract.x * inv_pix_fract.y;
	weights[1] = pix_fract.x * inv_pix_fract.y;
	weights[2] = inv_pix_fract.x * pix_fract.y;
	weights[3] = pix_fract.x * pix_fract.y;

	float sum_weight = 0;
	float3 sum_val = 0.0;
	float sum_spp = 0.0;
	float sum_moment = 0.0;
    
	for(int i = 0; i < 4; i++) {
	    int2 p = prev_frame_pixel + offsets[i];

	    int linear_p = linear(p);

	    if(p.x < 0 || p.y < 0 ||
	       p.x >= IMAGE_WIDTH || p.y >= IMAGE_HEIGHT) {
		continue;
	    }

	
	    float3 prev_wp = load_float3(prev_positions, linear_p);
	    float3 pos_diff = prev_wp - pos_load;
	    float ps_diff_squared = dot(pos_diff, pos_diff);

	    if(ps_diff_squared >= POSITION_TOLERANCE) {
		continue;
	    }

	    float3 prev_normal_load = load_float3(prev_normals, linear_p);
	    float3 n_dist = normal_load - prev_normal_load;

	    if(dot(n_dist, n_dist) >= NORMAL_TOLERANCE) {
		continue;
	    }

	    float3 val = load_float3(prev_accumulated, linear_p);
	    float3 prev_ut_val = load_float3(prev_util, linear_p);

	    sum_val += weights[i] * val;
	    sum_spp += weights[i] * prev_ut_val.x;
	    sum_moment += weights[i] * prev_ut_val.y;
	    sum_weight += weights[i];
	}

	if(sum_weight > 0.0) {
	    sum_spp /= sum_weight;
	    sum_val /= sum_weight;
	    sum_moment /= sum_weight;
	}
	
	float blend_a = max(1.0 / (sum_spp + 1.0), 0.15);
	float moment_a = max(1.0 / (sum_spp + 1.0), 0.2);

	float new_spp = sum_spp + 1.0;
	float new_moment = (1 - moment_a) * sum_moment + moment_a * pow(luminance(noise_load), 2.0f);

	acc_out = (1 - blend_a) * sum_val + blend_a * noise_load;

	util_val = (float3){new_spp, new_moment, linear_z};
    }

    store_float3(curr_accumulated, linear_pixel, acc_out);
    store_float3(curr_util, linear_pixel, util_val);
}

__kernel void compute_variance(const __global float* restrict curr_normals,
			       const __global float* restrict curr_accumulated,
			       const __global float* restrict curr_util,
			       __global float* restrict output_image) {
    
    const float NORMAL_PHI = 1e-2;
    // const float POSITION_PHI = 1e0; // Should depend on depth, but oh well
    const float COLOR_PHI = 1.0e1;

    const int2 gid = {get_global_id(0), get_global_id(1)};
    
    const int linear_pixel = linear(gid);

    float3 normal_load = load_float3(curr_normals, linear_pixel);
    float3 acc_load = load_float3(curr_accumulated, linear_pixel);
    float3 util_load = load_float3(curr_util, linear_pixel);

    float curr_lum = luminance(acc_load);

    float spp = util_load.x;
    float moment = util_load.y;
    float linear_z = util_load.z;

    float phiDepth = max(5e-3, 1e-8) * 3.0;

    float sum_weights = 0.0;
    float sum_moment = 0.0;
    float3 sum_accum = 0.0;

    float variance;
    
    if(spp < 4.0) {
	const int radius = 3;

	for(int yy = -radius; yy <= radius; yy++) {
	    for(int xx = -radius; xx <= radius; xx++) {
		const int2 p = gid + (int2){xx, yy};

		if(p.x < 0 || p.y < 0 ||
		   p.x >= IMAGE_WIDTH || p.y >= IMAGE_HEIGHT) {
		    continue;
		}

		int lp = linear(p);
		
		float3 local_util = load_float3(curr_util, lp);
		
		float local_moment = local_util.y;
		float local_z = local_util.z;

		float3 local_normal = load_float3(curr_normals, lp);
		float3 local_accum = load_float3(curr_accumulated, lp);

		float wnorm = pow(max(dot(local_normal, normal_load), 0.0f), NORMAL_PHI);
		float wpos = (xx == 0 && yy == 0) ? 0.0 : fabs(local_z - linear_z) / (phiDepth * length(convert_float2(((int2){xx, yy}))));
		float wcolor = fabs(curr_lum - luminance(local_accum)) / COLOR_PHI;

		float weight = exp(- wcolor - wpos - wnorm);
		sum_weights += weight;
		sum_moment += weight * moment;
		sum_accum += weight * local_accum;
	    }
	}

	sum_weights = fmax(sum_weights, 1e-5f);
	sum_moment /= sum_weights;
	sum_accum /= sum_weights;
    
	variance = sum_moment - pown(luminance(sum_accum), 2);
	variance *= 4.0 / spp;
    } else {
	variance = moment - curr_lum * curr_lum;
	sum_accum = acc_load;
    }

    store_float4(output_image, linear_pixel, (float4){sum_accum.x, sum_accum.y, sum_accum.z, variance}); 
}


// util for atrous
float variance_center(int2 pixel_coords,
		      const __global float* restrict curr_accumulated) {
    const float kern[3] = 
	{ 1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0 };

    const int radius = 1;
    float sum = 0.0;
    for(int yy = -1; yy <= radius; yy++) {
	for(int xx = -1; xx <= radius; xx++) {
	    int2 p = pixel_coords + (int2){xx, yy};
	    
	    if(p.x < 0 || p.y || p.x >= IMAGE_WIDTH || p.y >= IMAGE_HEIGHT) {
		continue;
	    }

	    int lp = linear(p);
	    float k = kern[abs(xx) + abs(yy)];

	    float4 acc_load = load_float4(curr_accumulated, lp);
	    sum += acc_load.w * k;
	}
    }

    return sum;
}

__kernel void atrous(const __global float* restrict curr_normals,
		     const __global float* restrict albedo,
		     const __global float* restrict curr_util,
		     const __global float* restrict curr_accumulated,
		     __global float* restrict output_image,
		     int step_size,
		     int last_time) {
    
    /* const float NORMAL_PHI = 3e-2;
    const float POSITION_PHI = 1.0; // Should depend on depth, but oh well
    const float COLOR_PHI = 10.0; */

    const float NORMAL_PHI = 1e-2;
    // const float POSITION_PHI = 3e0; // Should depend on depth, but oh well
    const float COLOR_PHI = 1.0e1;
    
    const float kernelWeights[3] = { 1.0, 2.0 / 3.0, 1.0 / 6.0 };

    const int2 gid = {get_global_id(0), get_global_id(1)};
    
    const int linear_pixel = linear(gid);
    
    float var = variance_center(linear_pixel, curr_accumulated);

    float3 util_load = load_float3(curr_util, linear_pixel);
    float3 normal_load = load_float3(curr_normals, linear_pixel);
    float4 acc_load = load_float4(curr_accumulated, linear_pixel);

    float3 curr_acc = acc_load.xyz;
    float variance = acc_load.w;
    
    float curr_luminance = luminance(curr_acc);

    float linear_z = util_load.z;
    
    float phi_color = COLOR_PHI * sqrt(max(0.0f, 1e-10f + var));
    float phiDepth = max(5e-3, 1e-8) * step_size;

    // Store center pixel with weight 1:
    float sum_weights = 1.0;
    float3 sum_accum = curr_acc;
    float sum_variance = variance;

    for(int yy = -2; yy <= 2; yy++) {
	for(int xx = -2; xx <= 2; xx++) {

	    int2 p = gid + (int2){xx, yy} * step_size;

	    if(p.x < 0 || p.y < 0 ||
	       p.x >= IMAGE_WIDTH || p.y >= IMAGE_HEIGHT ||
	       (xx == 0 && yy == 0)) {
		continue;
	    }

	    int lp = linear(p);

	    float4 local_acc = load_float4(curr_accumulated, lp);
	    float3 local_util = load_float3(curr_util, lp);
	    float3 local_normal = load_float3(curr_normals, lp);

	    float local_z = local_util.z;
	    float3 local_acc_color = (float3){local_acc.x, local_acc.y, local_acc.z};
	    float local_luminance = luminance(local_acc_color);

	    float wnorm = pow(max(dot(local_normal, normal_load), 0.0f), NORMAL_PHI);
	    float wpos = fabs(local_z - linear_z) / (phiDepth * length(convert_float2(((int2){xx, yy}))));
	    float wcolor = fabs(local_luminance - curr_luminance) / COLOR_PHI;

	    float weight = exp(-wcolor - wpos - wnorm);

	    float kx = kernelWeights[abs(xx)];
	    float ky = kernelWeights[abs(yy)];

	    weight *= kx * ky;

	    float local_var = local_acc.w;

	    sum_weights += weight;
	    sum_accum += weight * local_acc_color;
	    sum_variance += weight * weight * local_var;
	}
    }

    // Yes, this is right, accourding to Schied et al. 2017
    float total_var = sum_variance / (sum_weights * sum_weights);
    float3 total_accum = sum_accum / sum_weights;

    if(last_time == 1) {
	float3 alb = load_float3(albedo, linear_pixel);

	float3 modulated = clamp(linear_to_srgb(alb * total_accum), 0.0f, 1.0f);
        
	store_float3(output_image, linear_pixel, modulated);
    } else {
	float4 res = (float4){total_accum.x, total_accum.y, total_accum.z, total_var};
	store_float4(output_image, linear_pixel, res);
    }
}
