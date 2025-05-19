#pragma once

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#include <CL/cl.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "CLUtil.hpp"
#include "SDKBitMap.hpp"

#define SAMPLE_VERSION "HSV-CONVERSION-1.0"
#define INPUT_IMAGE "/input"
#define OUTPUT_IMAGE "outputHSV.bmp"
#define OUTPUT_IMAGE_TIF "outputHSV.tif"
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
    cl::Buffer outputImageBuffer;
    SDKBitMap inputBitmap;
    cv::Mat imageRGB;
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
    bool readBmp;
    std::string selectedDevice;
    size_t selectedPlatform;

public:
    Histogram(size_t H_BINS, size_t S_BINS, bool outHsv, bool readBmp,  size_t selectedPlatform, std::string selectedDevice)
        : inputImageData(NULL), outputImageData(NULL),
          byteRWSupport(true), H_BINS(H_BINS), S_BINS(S_BINS),
          outHsv(outHsv), readBmp(readBmp), selectedDevice(selectedDevice), 
          selectedPlatform(selectedPlatform)
    {
        pixelSize = sizeof(uchar4);
        pixelData = NULL;
    }

    ~Histogram() {}

    int readInputImage();
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

private:
    int writeToConsoleAvailabeComputeDevice();
    int writeToConsoleAvailabeDevice();
    int readInputImageTiff(const std::string& path);
    int readInputImageBmp(const std::string& path);
};
