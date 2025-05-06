#pragma once

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
#define OUTPUT_IMAGE "outputHSV.bmp"
#define OUTPUT_HIST "outputHist2D.bmp"

using namespace appsdk;

class Histogram
{
    cl_uchar4* inputImageData;
    cl_uchar4* outputImageData;
    std::vector<unsigned int> outputBufferHistData;

    cl::Context context;
    std::vector<cl::Device> devices;
    cl::Device device;
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
    size_t maxWorkGroupSizeForDevice;
    size_t maxWorkGroupSizeForKernel;
    cl_uint H_BINS;
    cl_uint S_BINS;
    bool outHsv;

public:
    CLCommandArgs* sampleArgs;

    Histogram(size_t H_BINS, size_t S_BINS, bool outHsv)
        : inputImageData(NULL), outputImageData(NULL), byteRWSupport(true), H_BINS(H_BINS), S_BINS(S_BINS), outHsv(outHsv)
    {
        sampleArgs = new CLCommandArgs();
        pixelSize = sizeof(uchar4);
        pixelData = NULL;
    }

    ~Histogram() {}

    int readInputImage(std::string inputImageName);
    int writeOutputImage(std::string outputImageName);
    int setupCL();
    int runCLKernels();
    int setup();
    int run();
    int cleanup();
    void setMaxNumberOfWorkGroup();
    auto countGlobalAndLocalSize();
    void createHistogramOutput();
    int saveHistogramAsImage(const std::string& filename);
};
