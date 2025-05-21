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
#include <filesystem>
#include <vector>

#define SAMPLE_VERSION "HSV-CONVERSION-1.0"
#define INPUT_IMAGE "input"
#define OUTPUT_IMAGE "outputHSV"
#define OUTPUT_HIST "outputHist2D"
#define OUTPUT_HIST_CSV "histogram_output"
#define INF_LOOP_DIR "Inputs"

using namespace appsdk;

class Histogram
{
    std::vector<cl_uchar4> inputImageData;
    std::vector<cl_uchar4> outputImageData;
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
    size_t maxWorkGroupSizeForDevice;
    size_t maxWorkGroupSizeForKernel;
    cl_uint H_BINS;
    cl_uint S_BINS;
    const bool outHsv;
    const bool readBmp;
    const bool histAsCsv;
    const bool histGray;
    const bool infLoop;
    std::string selectedDevice;
    size_t selectedPlatform;
    std::vector<std::string> fileNames;
    bool firstLoop = true;
    uint64_t indexOfInput{0};
    std::vector<double> timesVector;
    std::array<cl::NDRange, 2> globalAndLocalSize;
    uint64_t iteration{0};
    const uint64_t maxIter;
    bool successSaveImg{false};
    bool successSaveHist{false};
    cl::Kernel visualizeKernel;

public:

    Histogram(size_t H_BINS, size_t S_BINS, const bool outHsv, const bool readBmp,  size_t selectedPlatform, std::string selectedDevice, const bool histAsCsv, const bool histGray, const bool infLoop,const uint64_t maxIter)
        : H_BINS(H_BINS), S_BINS(S_BINS),
          outHsv(outHsv), readBmp(readBmp), selectedDevice(selectedDevice), 
          selectedPlatform(selectedPlatform), histAsCsv(histAsCsv),
          histGray(histGray), infLoop(infLoop), maxIter(maxIter)
    {
        pixelSize = sizeof(uchar4);
        pixelData = NULL;
    }

    ~Histogram()
    {
        writeSavedItems();
        writeDurationTime();
    }

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
    std::string createOutputFielname(std::string&& firstPath);
    void writeSavedItems();
    void writeDurationTimeForIteration(double time);
    std::string createPathToInput();
    void writeDurationTime();
    void readFileNames();
    void writeToConsoleSelectedComputeDevice();
    int writeToConsoleAvailabeComputeDevice();
    int writeToConsoleAvailabeDevice();
    void writeToConsoleSelectedDevice();
    int readInputImageTiff(const std::string& path);
    int readInputImageBmp(const std::string& path);
    void saveBufferToCSV2D(cl::CommandQueue& queue, cl::Buffer& buffer, size_t H, size_t S, const std::string& filename);
};
