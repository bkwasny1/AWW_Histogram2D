#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "Histogram.hpp"
#include "cmath"
#include "map"
#include "array"
#include <getopt.h>
#include <algorithm>
#include "stb_image_write.h"
#include "Converter.hpp"
#include <chrono>


int Histogram::readInputImage(std::string inputImageName)
{
    inputBitmap.load(inputImageName.c_str());
    if (!inputBitmap.isLoaded())
    {
        std::cout << "Failed to load input image!" << std::endl;
        return SDK_FAILURE;
    }

    height = inputBitmap.getHeight();
    width = inputBitmap.getWidth();

    inputImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    outputImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    memset(outputImageData, 0, width * height * pixelSize);

    pixelData = inputBitmap.getPixels();
    if (!pixelData) return SDK_FAILURE;
    memcpy(inputImageData, pixelData, width * height * pixelSize);

    return SDK_SUCCESS;
}

void Histogram::createHistogramOutput()
{
    H_BINS = (H_BINS == 0) ? 180 : H_BINS;
    S_BINS = (S_BINS == 0) ? 256 : S_BINS;
    outputBufferHistData.resize(H_BINS*S_BINS);
}

int Histogram::writeOutputImage(std::string outputImageName)
{
    memcpy(pixelData, outputImageData, width * height * pixelSize);
    return inputBitmap.write(outputImageName.c_str()) ? SDK_SUCCESS : SDK_FAILURE;
}

void Histogram::setMaxNumberOfWorkGroup()
{
    device.getInfo(CL_DEVICE_MAX_WORK_GROUP_SIZE, &maxWorkGroupSizeForDevice);
    kernel.getWorkGroupInfo(device, CL_KERNEL_WORK_GROUP_SIZE, &maxWorkGroupSizeForKernel);
    std::clog<<"Maksymalna liczba work itemow dla urzadzenia: "<<maxWorkGroupSizeForDevice<<", dla kernela: "<<maxWorkGroupSizeForKernel<<std::endl;
}

auto Histogram::countGlobalAndLocalSize()
{
    size_t localSizeKernel = static_cast<size_t>(std::sqrt(maxWorkGroupSizeForKernel));
    size_t globalX = ((width + (localSizeKernel-1))/localSizeKernel)*localSizeKernel;
    size_t globalY = ((height + (localSizeKernel-1))/localSizeKernel)*localSizeKernel;
    cl::NDRange global(globalX, globalY);
    cl::NDRange local(localSizeKernel, localSizeKernel);
    std::array<cl::NDRange, 2> result{global,local};
    printf("GlobalSize: (%d,%d), LocalSize: (%d,%d)\n",globalY,globalX,localSizeKernel,localSizeKernel);
    return result;

}

int Histogram::saveHistogramAsImage(const std::string& filename)
{
    const int width = S_BINS;
    const int height = H_BINS;
    const int channels = 3;
    unsigned int maxValue = *std::max_element(outputBufferHistData.begin(), outputBufferHistData.end());

    std::vector<unsigned char> image(width * height * channels);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            float value = static_cast<float>(outputBufferHistData[index]) / maxValue;
            unsigned char red, green, blue;
            Converter::valueToParulaColor(value, red, green, blue);
            image[(y * width + x) * channels + 0] = red;
            image[(y * width + x) * channels + 1] = green;
            image[(y * width + x) * channels + 2] = blue;
        }
    }

    return stbi_write_bmp(filename.c_str(), width, height, channels, image.data());
}

int Histogram::setupCL()
{
    cl_int err;
    cl::Platform::get(&platforms);
    if(platforms.empty()){
        std::cerr << "Brak dostępnych platform OpenCL!" << std::endl;
        return SDK_FAILURE;
    }
    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
    context = cl::Context(CL_DEVICE_TYPE_GPU, cps, NULL, NULL, &err);
    devices = context.getInfo<CL_CONTEXT_DEVICES>();
    if (devices.empty()) {
        std::cerr << "Brak urządzeń OpenCL!" << std::endl;
        return SDK_FAILURE;
    }
    
    device = devices[0];
    commandQueue = cl::CommandQueue(context, device, 0, &err);
    inputImage2D = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), width, height);
    outputImage2D = cl::Image2D(context, CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), width, height);

    SDKFile kernelFile;
    std::string kernelPath = getPath() + "Histogram_Kernels.cl";
    kernelFile.open(kernelPath.c_str());
    cl::Program::Sources source(1, std::make_pair(kernelFile.source().data(), kernelFile.source().size()));
    program = cl::Program(context, source);
    program.build(devices);
    kernel = cl::Kernel(program, "histogram2D");
    
    histogramBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, H_BINS * S_BINS * sizeof(cl_uint), outputBufferHistData.data());
    setMaxNumberOfWorkGroup();

    return SDK_SUCCESS;
}

int Histogram::runCLKernels()
{
    cl::size_t<3> origin;
    origin[0] = 0;
    origin[1] = 0;
    origin[2] = 0;

    cl::size_t<3> region;
    region[0] = width;
    region[1] = height;
    region[2] = 1;

    auto start = std::chrono::high_resolution_clock::now();
    commandQueue.enqueueWriteImage(inputImage2D, CL_TRUE, origin, region, 0, 0, inputImageData);
    kernel.setArg(0, inputImage2D);
    kernel.setArg(1, outputImage2D);
    kernel.setArg(2, histogramBuffer);
    kernel.setArg(3, width);
    kernel.setArg(4, height);
    kernel.setArg(5, H_BINS);
    kernel.setArg(6, S_BINS);
    auto globalAndLocalSize = countGlobalAndLocalSize();
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, globalAndLocalSize.at(0), globalAndLocalSize.at(1));
    commandQueue.enqueueReadImage(outputImage2D, CL_TRUE, origin, region, 0, 0, outputImageData);
    commandQueue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, outputBufferHistData.size() * sizeof(cl_uint), outputBufferHistData.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "\n\nCzas wykonania: " << elapsed.count() << " ms\n"<<std::endl;
    return SDK_SUCCESS;
}

int Histogram::setup()
{
    std::string path = getPath() + INPUT_IMAGE;
    createHistogramOutput();
    return readInputImage(path) == SDK_SUCCESS && setupCL() == SDK_SUCCESS ? SDK_SUCCESS : SDK_FAILURE;
}

int Histogram::run()
{
    return runCLKernels() == SDK_SUCCESS && writeOutputImage(OUTPUT_IMAGE) == SDK_SUCCESS && saveHistogramAsImage(OUTPUT_HIST) == SDK_SUCCESS ? SDK_SUCCESS : SDK_FAILURE;
}

int Histogram::cleanup()
{
    free(inputImageData);
    free(outputImageData);
    return SDK_SUCCESS;
}

struct ParsedArgs
{
    size_t H_BINS = 0;
    size_t S_BINS = 0;
    bool outHsv = false;
};

int parseArgument(ParsedArgs& parseArgs, int argc, char * argv[])
{
    struct option long_options[] = {
        {"H_BINS", required_argument, nullptr, 'H'},
        {"S_BINS", required_argument, nullptr, 'S'},
        {"HSV", no_argument, nullptr, 'r'},
        {nullptr, 0, nullptr, 0}         
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "H:S:r", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'H':
                parseArgs.H_BINS = std::stoi(optarg);
                std::cout << "H_BINS: " << parseArgs.H_BINS << std::endl;
                break;
            case 'S':
                parseArgs.S_BINS = std::stoi(optarg);
                std::cout << "S_BINS: " << parseArgs.S_BINS << std::endl;
                break;
            case 'r':
                parseArgs.outHsv = true;
                std::cout << "HSV opcja wlaczona." << std::endl;
                break;
            case '?':
                std::cerr << "Nieznana opcja lub brak wymaganych argumentow." << std::endl;
                return SDK_FAILURE;
            default:
                std::cerr << "Error podczas parsowania argumentow." << std::endl;
                return SDK_FAILURE;
        }
    }
    return SDK_SUCCESS;
}

int main(int argc, char * argv[])
{
    ParsedArgs parsedArgs;
    if (parseArgument(parsedArgs, argc, argv) != SDK_SUCCESS) return SDK_FAILURE;
    Histogram clHistogram{parsedArgs.H_BINS, parsedArgs.S_BINS,parsedArgs.outHsv};
    if (clHistogram.setup() != SDK_SUCCESS) return SDK_FAILURE;
    if (clHistogram.run() != SDK_SUCCESS) return SDK_FAILURE;
    if (clHistogram.cleanup() != SDK_SUCCESS) return SDK_FAILURE;
    return SDK_SUCCESS;
}
