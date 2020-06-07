#pragma once

#include "svgf.hpp"

#include <vector>

struct ImageData {
  std::vector<cl_float> out_data[NUM_FRAMES];
  std::vector<cl_float> albedos[NUM_FRAMES];
  std::vector<cl_float> normals[NUM_FRAMES];
  std::vector<cl_float> positions[NUM_FRAMES];
  std::vector<cl_float> noisy_input[NUM_FRAMES];
};

ImageData initializeData();


// (Copied from BMFR)

// Creates two same buffers and swap() call can be used to change which one is considered
// current and which one previous
template <class T>
class Double_buffer
{
    private:
        T a, b;
        bool swapped;

    public:
        template <typename... Args>
        Double_buffer(Args... args) : a(args...), b(args...), swapped(false){};
        T *current() { return swapped ? &a : &b; }
        T *previous() { return swapped ? &b : &a; }
        void swap() { swapped = !swapped; }
};

struct Operation_result
{
    bool success;
    std::string error_message;
    Operation_result(bool success, const std::string &error_message = "") :
        success(success), error_message(error_message) {}
};

Operation_result load_image(cl_float *image, const std::string file_name, const int frame);
Operation_result read_image_file(const std::string &file_name, const int frame, float *buffer);

void writeOutputImages(const ImageData& image_data, const std::string& output_file);

const char *getErrorString(cl_int error);
