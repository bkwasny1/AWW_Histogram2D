#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "Histogram.hpp"
#include "cmath"
#include "map"
#include "array"
#include <getopt.h>
#include <algorithm>
#include "stb_image_write.h"
#include <chrono>
#include <optional>

int Histogram::readInputImageTiff(const std::string& path)
{
    std::string inputImageName = path + std::string(".tif");
    cv::Mat image = cv::imread(inputImageName, cv::IMREAD_UNCHANGED);
    cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGBA);
    if (image.empty()) {
        return SDK_FAILURE;
    }
    height = image.rows;
    width = image.cols;
    return SDK_SUCCESS;
}

int Histogram::readInputImageBmp(const std::string& path)
{
    std::string inputImageName = path + std::string(".bmp");
    inputBitmap.load(inputImageName.c_str());
    if (!inputBitmap.isLoaded())
    {
        return SDK_FAILURE;
    }

    height = inputBitmap.getHeight();
    width = inputBitmap.getWidth();
    pixelData = inputBitmap.getPixels();
    if (!pixelData) return SDK_FAILURE;
    return SDK_SUCCESS;
}


int Histogram::readInputImage()
{

    std::string path = getPath() + INPUT_IMAGE;
    if(not readBmp)
    {
        if(readInputImageTiff(path) == SDK_FAILURE) return SDK_FAILURE;
    }
    else
    {
        if(readInputImageBmp(path) == SDK_FAILURE) return SDK_FAILURE;
    }
    
    inputImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
    if(outHsv)
    {
        outputImageData = (cl_uchar4*)malloc(width * height * sizeof(cl_uchar4));
        memset(outputImageData, 0, width * height * pixelSize);
    }

    if(not readBmp)
    {
        std::memcpy(inputImageData, imageRGB.data, width * height * sizeof(cl_uchar4));
    }
    else
    {
        memcpy(inputImageData, pixelData, width * height * pixelSize);
    }
    

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
    if(not outHsv)
    {
        return SDK_EXPECTED_FAILURE;
    }

    if(not readBmp)
    {
        cv::Mat output(height, width, CV_8UC4, outputImageData);
        cv::cvtColor(output, output, cv::COLOR_RGBA2BGRA);
        return cv::imwrite(OUTPUT_IMAGE_TIF, output) ? SDK_SUCCESS : SDK_FAILURE;
    }
    else
    {
        memcpy(pixelData, outputImageData, width * height * pixelSize);
        return inputBitmap.write(outputImageName.c_str()) ? SDK_SUCCESS : SDK_FAILURE;
    }
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
    printf("GlobalSize: (%d,%d), LocalSize: (%d,%d)\n\n",globalX, globalY,localSizeKernel,localSizeKernel);
    printf("Rozmiar obrazu: %dx%d\n", width, height);
    return result;

}
int Histogram::saveHistogramAsImage(const std::string& filename)
{
    cl_int err;
    unsigned int maxValue = *std::max_element(outputBufferHistData.begin(), outputBufferHistData.end());
    
    cl::Kernel visualizeKernel(program, "visualizeHistogram", &err);
    visualizeKernel.setArg(0, histogramBuffer);
    visualizeKernel.setArg(1, outputImageBuffer);
    visualizeKernel.setArg(2, S_BINS);
    visualizeKernel.setArg(3, H_BINS);
    visualizeKernel.setArg(4, maxValue);
    
    cl::NDRange global(S_BINS, H_BINS);
    commandQueue.enqueueNDRangeKernel(visualizeKernel, cl::NullRange, global);
    std::vector<unsigned char> image(S_BINS * H_BINS * 3);
    commandQueue.enqueueReadBuffer(outputImageBuffer, CL_TRUE, 0, image.size(), image.data());
    
    return (stbi_write_bmp(filename.c_str(), S_BINS, H_BINS, 3, image.data())) ? SDK_SUCCESS : SDK_FAILURE;
}

void Histogram::writeToConsoleSelectedComputeDevice()
{
    std::clog <<"Wybrano: "<< selectedDevice<<std::endl<< std::endl;
}

int Histogram::writeToConsoleAvailabeComputeDevice()
{
    std::clog<<std::endl;
    std::clog<<"Dostepne urzadzenia obliczeniowe: "<<std::endl;
    std::clog <<"  - GPU"<< std::endl;
    std::clog <<"  - CPU"<< std::endl<< std::endl;
    if((selectedDevice != "GPU") and (selectedDevice != "CPU"))
    {
        std::cerr<<"ERROR: Brak wybranego urzadzenia: dostepne CPU oraz GPU"<< std::endl;
        return SDK_FAILURE;
    } 
    return SDK_SUCCESS;
}
void Histogram::writeToConsoleSelectedDevice()
{
    std::clog << "Wybrano urzadzenie o nazwie: " << device.getInfo<CL_DEVICE_NAME>() << std::endl<< std::endl;
}

int Histogram::writeToConsoleAvailabeDevice()
{
    bool isSelected{false};
    std::map<std::string, uint8_t> version{{"GPU", CL_DEVICE_TYPE_GPU}, {"CPU", CL_DEVICE_TYPE_CPU}};
    for(const auto& [key, value] : version)
    {
        std::clog<<"Dostepne urzadzenia "<<key<<": "<<std::endl;
        size_t i{0};
        for (const auto& platform : platforms) {
            std::vector<cl::Device> allDevices;
            platform.getDevices(value, &allDevices);
            
            for (const auto& dev : allDevices) {
                std::string deviceName = dev.getInfo<CL_DEVICE_NAME>();
                std::clog << "  " << i << ": " << deviceName << std::endl;
                if(selectedDevice == key and selectedPlatform == i)
                {
                    devices = allDevices;
                    device = dev;
                    context = cl::Context(device);
                    isSelected = true;
                }
                i++;
            }
        }
        std::clog << std::endl;
    }
    if(isSelected) return SDK_SUCCESS;
    std::cerr << "ERROR: Brak urzadzenia "<<selectedDevice<<" o numerze: "<<selectedPlatform<< std::endl;
    return SDK_FAILURE;
}

int Histogram::setupCL()
{
    cl_int err;
    cl::Platform::get(&platforms);
    if(platforms.empty()){
        std::cerr << "ERROR: Brak dostepnych platform OpenCL!" << std::endl;
        return SDK_FAILURE;
    }
    
    if(writeToConsoleAvailabeComputeDevice() == SDK_FAILURE) return SDK_FAILURE;
    writeToConsoleSelectedComputeDevice();
    if(writeToConsoleAvailabeDevice() == SDK_FAILURE) return SDK_FAILURE;
    writeToConsoleSelectedDevice();
    

    commandQueue = cl::CommandQueue(context, device, 0, &err);
    inputImage2D = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), width, height);
    if(outHsv)
    {
        outputImage2D = cl::Image2D(context, CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), width, height);
    }

    SDKFile kernelFile;
    std::string kernelPath = getPath() + "Histogram_Kernels.cl";
    kernelFile.open(kernelPath.c_str());
    cl::Program::Sources source(1, std::make_pair(kernelFile.source().data(), kernelFile.source().size()));
    program = cl::Program(context, source);
    program.build(devices);
    std::string kernelName = "histogram2D";
    if(not outHsv)
    {
        kernelName = "histogram2DWithOutHsv";
    }
    kernel = cl::Kernel(program, kernelName.c_str());
    
    histogramBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, H_BINS * S_BINS * sizeof(cl_uint), outputBufferHistData.data());
    outputImageBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, S_BINS * H_BINS * 3 * sizeof(cl_uchar));
    setMaxNumberOfWorkGroup();
    

    return SDK_SUCCESS;
}

int Histogram::runCLKernels()
{
    size_t idxArg = 1;
    if(not outHsv)
    {
        idxArg = 0;
    }

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
    if(outHsv)
    {
        kernel.setArg(1, outputImage2D);
    }
    kernel.setArg(idxArg + 1, histogramBuffer);
    kernel.setArg(idxArg + 2, width);
    kernel.setArg(idxArg + 3, height);
    kernel.setArg(idxArg + 4, H_BINS);
    kernel.setArg(idxArg + 5, S_BINS);

    auto globalAndLocalSize = countGlobalAndLocalSize();
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, globalAndLocalSize.at(0), globalAndLocalSize.at(1));
    if(outHsv)
    {
        commandQueue.enqueueReadImage(outputImage2D, CL_TRUE, origin, region, 0, 0, outputImageData);
    }
    commandQueue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, outputBufferHistData.size() * sizeof(cl_uint), outputBufferHistData.data());
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "\n\nCzas wykonania: " << elapsed.count() << " ms\n"<<std::endl;
    return SDK_SUCCESS;
}

void Histogram::saveBufferToCSV2D(cl::CommandQueue& queue, cl::Buffer& buffer, size_t H, size_t S, const std::string& filename) {
    std::vector<int> hostHistogram(H * S);
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, hostHistogram.size() * sizeof(int), hostHistogram.data());

    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "ERROR: Nie można otworzyć pliku CSV: " << filename << std::endl;
        return;
    }

    for (size_t h = 0; h < H; ++h) {
        for (size_t s = 0; s < S; ++s) {
            outFile << hostHistogram[h * S + s];
            if (s < S - 1) outFile << ",";
        }
        outFile << "\n";
    }

    outFile.close();
    std::clog << "Zapisano histogram jako plik CSV: " << filename << std::endl;
}

int Histogram::setup()
{
    
    createHistogramOutput();
    return readInputImage() == SDK_SUCCESS && setupCL() == SDK_SUCCESS ? SDK_SUCCESS : SDK_FAILURE;
}

int Histogram::run()
{
    bool success = runCLKernels() == SDK_SUCCESS;

    if (success) {
        if(writeOutputImage(OUTPUT_IMAGE) == SDK_SUCCESS)
        {
            if(readBmp)
            {
                std::clog << "Zapisano obraz HSV: " << OUTPUT_IMAGE << std::endl; 
            }
            else
            {
                std::clog << "Zapisano obraz HSV: " << OUTPUT_IMAGE_TIF << std::endl; 
            }
        }
        if(saveHistogramAsImage(OUTPUT_HIST) == SDK_SUCCESS)
        {
            std::clog << "Zapisano histogram jako obraz: " << OUTPUT_HIST << std::endl; 
        };
        if(histAsCsv)
        {
            saveBufferToCSV2D(commandQueue, histogramBuffer, H_BINS, S_BINS, OUTPUT_HIST_CSV);
        }
    }

    return success ? SDK_SUCCESS : SDK_FAILURE;
}

int Histogram::cleanup()
{
    free(inputImageData);
    if(outHsv)
    {
        free(outputImageData);
    }
    return SDK_SUCCESS;
}

struct ParsedArgs
{
    size_t H_BINS = 0;
    size_t S_BINS = 0;
    size_t selectedPlatform = 0;
    std::string selectedDevice = "GPU";
    bool outHsv = false;
    bool readBmp = false;
    bool histAsCsv = false;
};

int parseArgument(ParsedArgs& parseArgs, int argc, char * argv[])
{
    struct option long_options[] = {
        {"H_BINS", required_argument, nullptr, 'H'},
        {"S_BINS", required_argument, nullptr, 'S'},
        {"device", required_argument, nullptr, 'D'},
        {"platform", required_argument, nullptr, 'P'},
        {"HSV", no_argument, nullptr, 'r'},
        {"BMP", no_argument, nullptr, 'B'},
        {"HistCSV", no_argument, nullptr, 'C'},
        {nullptr, 0, nullptr, 0}         
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "H:S:D:P:rBC", long_options, nullptr)) != -1) {
        switch (opt) {
            case 'H':
                parseArgs.H_BINS = std::stoi(optarg);
                break;
            case 'S':
                parseArgs.S_BINS = std::stoi(optarg);
                break;
            case 'P':
                parseArgs.selectedPlatform = std::stoi(optarg);
                break;
            case 'D':
                parseArgs.selectedDevice = std::string(optarg);
                break;
            case 'r':
                parseArgs.outHsv = true;
                break;
            case 'B':
                parseArgs.readBmp = true;
                break;
            case 'C':
                parseArgs.histAsCsv = true;
                break;
            case '?':
                std::cerr << "ERROR: Nieznana opcja lub brak wymaganych argumentow." << std::endl;
                return SDK_FAILURE;
            default:
                std::cerr << "ERROR: Blad podczas parsowania argumentow." << std::endl;
                return SDK_FAILURE;
        }
    }
    return SDK_SUCCESS;
}



int main(int argc, char * argv[])
{
    ParsedArgs parsedArgs;
    if (parseArgument(parsedArgs, argc, argv) != SDK_SUCCESS) return SDK_FAILURE;
    Histogram clHistogram{parsedArgs.H_BINS, parsedArgs.S_BINS,parsedArgs.outHsv, parsedArgs.readBmp, 
                            parsedArgs.selectedPlatform, parsedArgs.selectedDevice, parsedArgs.histAsCsv};
    if (clHistogram.setup() != SDK_SUCCESS) return SDK_FAILURE;
    if (clHistogram.run() != SDK_SUCCESS) return SDK_FAILURE;
    if (clHistogram.cleanup() != SDK_SUCCESS) return SDK_FAILURE;
    return SDK_SUCCESS;
}
