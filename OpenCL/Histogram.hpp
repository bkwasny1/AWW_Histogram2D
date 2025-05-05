#ifndef HISTOGRAM_H_
#define HISTOGRAM_H_

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"
#include "SDKBitMap.hpp"

#define SAMPLE_VERSION "HSV-CONVERSION-1.0"
#define INPUT_IMAGE "input.bmp"
#define OUTPUT_IMAGE "output.bmp"
#define GROUP_SIZE 256

#define H_BINS 180
#define S_BINS 256

using namespace appsdk;

class Histogram
{
    cl_uchar4* inputImageData;
    cl_uchar4* outputImageData;

    cl::Context context;
    std::vector<cl::Device> devices;
    std::vector<cl::Device> device;
    std::vector<cl::Platform> platforms;
    cl::Image2D inputImage2D;
    cl::Image2D outputImage2D;
    cl::CommandQueue commandQueue;
    cl::Program program;
    cl::Kernel kernel;
    cl::Buffer histogramBuffer;
    SDKBitMap inputBitmap;
    uchar4* pixelData;
    cl_uint pixelSize;
    cl_uint width;
    cl_uint height;
    cl_bool byteRWSupport;
    size_t kernelWorkGroupSize;
    size_t blockSizeX;
    size_t blockSizeY;
    int iterations;
    int imageSupport;

public:
    CLCommandArgs* sampleArgs;

    Histogram()
        : inputImageData(NULL), outputImageData(NULL), byteRWSupport(true)
    {
        sampleArgs = new CLCommandArgs();
        pixelSize = sizeof(uchar4);
        pixelData = NULL;
        blockSizeX = GROUP_SIZE;
        blockSizeY = 1;
        iterations = 1;
        imageSupport = 0;
    }

    ~Histogram() {}

    int readInputImage(std::string inputImageName);
    int writeOutputImage(std::string outputImageName);
    int setupCL();
    int runCLKernels();
    int setup();
    int run();
    int cleanup();
};

#endif // HISTOGRAM_H_