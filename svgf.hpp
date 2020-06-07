#pragma once

#include "CLUtils/CLUtils.hpp"



#define _CRT_SECURE_NO_WARNINGS
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

// ### Choose your OpenCL device and platform with these defines ###
#define PLATFORM_INDEX 0
#define DEVICE_INDEX 0

// Location where input frames and feature buffers are located
// #define INPUT_DATA_PATH /home/haakon/data/bmfr_data/sponza-(static-camera)/inputs
#define INPUT_DATA_PATH /home/haakon/Documents/NTNU/TDT4900/dataconstruction/output

#define INPUT_DATA_PATH_STR STR(INPUT_DATA_PATH)
// camera_matrices.h is expected to be in the same folder
#include STR(INPUT_DATA_PATH/camera_matrices.h)
// These names are appended with NN.exr, where NN is the frame number
#define NOISY_FILE_NAME INPUT_DATA_PATH_STR"/color"
#define NORMAL_FILE_NAME INPUT_DATA_PATH_STR"/shading_normal"
#define POSITION_FILE_NAME INPUT_DATA_PATH_STR"/world_position"
#define ALBEDO_FILE_NAME INPUT_DATA_PATH_STR"/albedo"
#define OUTPUT_FILE_NAME "outputs/output"

const int IMAGE_WIDTH = 1280;
const int IMAGE_HEIGHT = 720;
const int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT;

const int LOCAL_SIZE_X = 32;
const int LOCAL_SIZE_Y  = 1;

const int NUM_FRAMES = 60; 

const int NUM_ATROUS_ITERATIONS = 4;
