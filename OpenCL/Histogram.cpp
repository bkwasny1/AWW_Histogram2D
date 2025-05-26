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
#include <numeric>
#include <iomanip>
#include "ReadKeyboard.hpp"

void Histogram::readFileNames(){
    std::string folderPath =  getPath()  + INF_LOOP_DIR;
    if (!std::filesystem::exists(folderPath) || !std::filesystem::is_directory(folderPath)) {
        std::cerr << "Nieznaleziono folderu Images" << std::endl;
        std::exit(SDK_FAILURE);
    }
    std::string extension = (readBmp) ? ".bmp" : ".tif";

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.is_regular_file() && entry.path().extension() == extension) {
            fileNames.push_back(entry.path().string());
        }
    }
    if (fileNames.size() == 0)
    {
        std::cerr << "Brak zdjec o rozszerzeniu "<<extension<<" w folderze Images" << std::endl;
        std::exit(SDK_FAILURE);
    }
}


int Histogram::readInputImageTiff(const std::string& path)
{
    if(readLoop)
    {
        cv::Mat image = cv::imread(path, cv::IMREAD_UNCHANGED);
        if (image.empty()) 
        {
            return SDK_FAILURE;
        }
        if(infLoop)
        {
            cv::cvtColor(image, image, cv::COLOR_BGR2RGBA);
            imageInputVectorTif.push_back(image);
            return SDK_SUCCESS;
        }
        else
        {
            cv::cvtColor(image, imageRGB, cv::COLOR_BGR2RGBA);
        }
    }
    else
    {
        imageRGB = imageInputVectorTif[indexOfInput];
    }
    
    height = imageRGB.rows;
    width = imageRGB.cols;
    return SDK_SUCCESS;
}

int Histogram::readInputImageBmp(const std::string& path)
{
    if(readLoop)
    {
        SDKBitMap image;
        image.load(path.c_str());
        if (!image.isLoaded())
        {
            return SDK_FAILURE;
        }
        if(infLoop)
        {
            imageInputVectorBmp.push_back(image);
            return SDK_SUCCESS;
        }
        else
        {
            inputBitmap = image;
        }
    }
    else
    {
        inputBitmap = imageInputVectorBmp[indexOfInput];
    }
    
    height = inputBitmap.getHeight();
    width = inputBitmap.getWidth();
    pixelData = inputBitmap.getPixels();
    if (!pixelData) return SDK_FAILURE;
    return SDK_SUCCESS;
}

std::string Histogram::createPathToInput()
{
    if(infLoop)
    {
        if(indexOfInput > fileNames.size() - 1){ indexOfInput = 0; }
        return fileNames[indexOfInput];
    }
    else
    {
        std::string extension = (readBmp) ? ".bmp" : ".tif";
        return static_cast<std::string>(getPath() + INPUT_IMAGE + extension);
    }
    
}

int Histogram::readInputImage()
{
    
    std::string path = createPathToInput();
    if(not readBmp)
    {
        if(readInputImageTiff(path) == SDK_FAILURE) return SDK_FAILURE;
    }
    else
    {
        if(readInputImageBmp(path) == SDK_FAILURE) return SDK_FAILURE;
    }
    if(infLoop and readLoop)
    {
        return SDK_SUCCESS;
    }

    inputImageData = std::vector<cl_uchar4>(width * height, cl_uchar4{0, 0, 0, 0});
    if(outHsv)
    {
        outputImageData = std::vector<cl_uchar4>(width * height, cl_uchar4{0, 0, 0, 0});
    }

    if(not readBmp)
    {
        std::memcpy(inputImageData.data(), imageRGB.data, width * height * sizeof(cl_uchar4));
    }
    else
    {
        std::memcpy(inputImageData.data(), pixelData, width * height * pixelSize);
    }
    return SDK_SUCCESS;
}

void Histogram::createHistogramOutput()
{
    H_BINS = (H_BINS == 0) ? 180 : H_BINS;
    S_BINS = (S_BINS == 0) ? 256 : S_BINS;
    outputBufferHistData = std::vector<unsigned int>(H_BINS*S_BINS, 0);
}

int Histogram::writeOutputImage(std::string outputImageName)
{
    if(not outHsv)
    {
        return SDK_EXPECTED_FAILURE;
    }

    if(not readBmp)
    {
        cv::Mat output(height, width, CV_8UC4, outputImageData.data());
        cv::cvtColor(output, output, cv::COLOR_RGBA2BGRA);
        return cv::imwrite(outputImageName, output) ? SDK_SUCCESS : SDK_FAILURE;
    }
    else
    {
        memcpy(pixelData, outputImageData.data(), width * height * pixelSize);
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
    if(not infLoop)
    {
        printf("GlobalSize: (%d,%d), LocalSize: (%d,%d)\n\n",globalX, globalY,localSizeKernel,localSizeKernel);
        printf("Rozmiar obrazu: %dx%d\n", width, height);
    }
    return result;

}
int Histogram::saveHistogramAsImage(const std::string& filename)
{
    cl_int err;
    unsigned int maxValue = *std::max_element(outputBufferHistData.begin(), outputBufferHistData.end());
    cl_uint flag = (histGray) ? 1 : 0;
    
    if(firstLoop)
    {
        visualizeKernel = cl::Kernel(program, "visualizeHistogram", &err);
        firstLoop = false;
    }
    visualizeKernel.setArg(0, histogramBuffer);
    visualizeKernel.setArg(1, outputImageBuffer);
    visualizeKernel.setArg(2, S_BINS);
    visualizeKernel.setArg(3, H_BINS);
    visualizeKernel.setArg(4, maxValue);
    visualizeKernel.setArg(5, flag);

    size_t localSizeKernel = static_cast<size_t>(std::sqrt(maxWorkGroupSizeForKernel));
    size_t globalX = ((S_BINS + (localSizeKernel-1))/localSizeKernel)*localSizeKernel;
    size_t globalY = ((H_BINS + (localSizeKernel-1))/localSizeKernel)*localSizeKernel;
    cl::NDRange global(globalX, globalY);
    cl::NDRange local(localSizeKernel, localSizeKernel);
    commandQueue.enqueueNDRangeKernel(visualizeKernel, cl::NullRange, global, local);

    std::vector<unsigned char> image(S_BINS * H_BINS * 3);
    commandQueue.enqueueReadBuffer(outputImageBuffer, CL_FALSE, 0, image.size(), image.data());
    commandQueue.finish();
    
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

void Histogram::writeDurationTime()
{   
    std::clog<<"\n\n" << std::fixed << std::setprecision(2);
    if(timesVector.size() == 1)
    {
        std::clog << "Czas wykonania kernela: " << timesVector[0] << " ms\n"<<std::endl;
    }
    else
    {
        double minVal = *std::min_element(timesVector.begin(), timesVector.end());
        double maxVal = *std::max_element(timesVector.begin(), timesVector.end());
        double avgVal = std::accumulate(timesVector.begin(), timesVector.end(), 0.0) / timesVector.size();
        std::clog << "Minimalny czas wykonania kernela: " << minVal << " ms\n"<<std::endl;
        std::clog << "Maksymalny czas wykonania kernela: " << maxVal << " ms\n"<<std::endl;
        std::clog << "Sredni czas wykonania kernela: " << avgVal << " ms\n"<<std::endl;
    }
}

void Histogram::writeSavedItems()
{
    if(not infLoop)
    {
        return;
    }
    std::clog<<"\n"<<std::endl;
    if(successSaveImg and outHsv)
    {
        
        if(readBmp)
        {
            std::clog << "Zapisano obrazy HSV w formacie .bmp"<< std::endl; 
        }
        else
        {
            std::clog << "Zapisano obrazy HSV w formacie .tif"<< std::endl; 
        }
        
    }

    if(successSaveHist)
    {
        std::clog << "Zapisano histogramy w formacie .bmp"<< std::endl;
    }

    if(histAsCsv)
    {
        std::clog << "Zapisano histogramy w formacie .csv" << std::endl;
    }

}

std::string Histogram::createOutputFielname(std::string&& firstPath)
{
    if(infLoop)
    {
        std::string result;
        if(firstPath == OUTPUT_HIST)
        {
            result =  firstPath + std::to_string(indexOfInput) + std::string(".bmp");
        }
        else if(firstPath == OUTPUT_IMAGE)
        {
            std::string extension = (readBmp) ? ".bmp" : ".tif";
            result =  firstPath + std::to_string(indexOfInput) + extension;
        }
        else
        {
            result = firstPath + std::to_string(indexOfInput) + std::string(".csv");
        }
        return result;
    }
    else
    {
        if(firstPath == OUTPUT_HIST)
        {
            return firstPath + ".bmp";
        }
        else if(firstPath == OUTPUT_IMAGE)
        {
            std::string extension = (readBmp) ? ".bmp" : ".tif";
            return firstPath + extension;
        }
        else
        {
            return firstPath + std::string(".csv");
        }
    }
}

void Histogram::writeDurationTimeForIteration(double time)
{
    if(iteration == 0)
    {
        if (infLoop and maxIter != 1)
        {
            std::clog<<"\n\n-------------- Liczba iteracji: "<<maxIter<<" --------------"<<std::endl;
        }
        else if (infLoop and maxIter == 1)
        {


            std::clog<<"\n\n Aby zatrzymac wykonywanie programu wcisnij klawisz q lub ESC"<<std::endl;
            std::clog<<"-------------- Liczba iteracji: inf --------------"<<std::endl;
        }
    }
    if(infLoop)
    {
        iteration++;
        std::clog<<"Czas wykonania kernela dla iteracji "<<iteration<<": "<<time<<std::endl;
    }
}

int Histogram::setupCL()
{
    if (firstLoop)
    {
        cl::Platform::get(&platforms);
        if(platforms.empty()){
            std::cerr << "ERROR: Brak dostepnych platform OpenCL!" << std::endl;
            return SDK_FAILURE;
        }
        
        if(writeToConsoleAvailabeComputeDevice() == SDK_FAILURE) return SDK_FAILURE;
        writeToConsoleSelectedComputeDevice();
        if(writeToConsoleAvailabeDevice() == SDK_FAILURE) return SDK_FAILURE;
        writeToConsoleSelectedDevice();
        cl_int err;
        commandQueue = cl::CommandQueue(context, device, 0, &err);
    
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
        setMaxNumberOfWorkGroup();
        histogramBuffer = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, H_BINS * S_BINS * sizeof(cl_uint), outputBufferHistData.data());
        outputImageBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, S_BINS * H_BINS * 3 * sizeof(cl_uchar));
    }
    inputImage2D = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), width, height);
    if(outHsv)
    {
        outputImage2D = cl::Image2D(context, CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), width, height);
    }
    
    return SDK_SUCCESS;
}

int Histogram::runCLKernels(std::string& name)
{
    if(infLoop)
    {
        commandQueue.enqueueFillBuffer(outputImageBuffer, 0, 0, H_BINS * S_BINS * sizeof(cl_uchar));
        commandQueue.enqueueFillBuffer(histogramBuffer, 0, 0, H_BINS * S_BINS * sizeof(cl_uint));
    }
    
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
    commandQueue.enqueueWriteImage(inputImage2D, CL_FALSE, origin, region, 0, 0, inputImageData.data());

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


    globalAndLocalSize = countGlobalAndLocalSize();

    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, globalAndLocalSize.at(0), globalAndLocalSize.at(1));
    
    if(outHsv)
    {
        commandQueue.enqueueReadImage(outputImage2D, CL_FALSE, origin, region, 0, 0, outputImageData.data());
    }
    commandQueue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, outputBufferHistData.size() * sizeof(cl_uint), outputBufferHistData.data());
    
    successSaveHist = saveHistogramAsImage(name) == SDK_SUCCESS;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;
    timesVector.push_back(elapsed.count());
    writeDurationTimeForIteration(elapsed.count());

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
    if(not infLoop)
    {
        std::clog << "Zapisano histogram jako plik CSV: " << filename << std::endl;
    }
}

int Histogram::setup()
{
    if (firstLoop)
    {
        createHistogramOutput();
        if (infLoop)
        {
            readFileNames();
            for(auto i = 0u; i < fileNames.size(); i++)
            {
                readInputImage();
                indexOfInput ++;
            }
            readLoop = false;
        }

    }
    return readInputImage() == SDK_SUCCESS && setupCL() == SDK_SUCCESS ? SDK_SUCCESS : SDK_FAILURE;
}


int Histogram::run()
{
    std::string name = createOutputFielname(OUTPUT_HIST);
    bool success = runCLKernels(name) == SDK_SUCCESS;
    
    if (success) {
        name = createOutputFielname(OUTPUT_IMAGE);
        successSaveImg = writeOutputImage(name) == SDK_SUCCESS;
        if(successSaveImg)
        {
            if(not infLoop)
            {
                std::clog << "Zapisano obraz HSV: " << name << std::endl; 
            }
            
        }
        if(successSaveHist)
        {
            if(not infLoop)
            {
                name = createOutputFielname(OUTPUT_HIST);
                std::clog << "Zapisano histogram jako obraz: " << name << std::endl;
            }
        };
        if(histAsCsv)
        {   
            name = createOutputFielname(OUTPUT_HIST_CSV);
            saveBufferToCSV2D(commandQueue, histogramBuffer, H_BINS, S_BINS, name);
        }
        indexOfInput ++;
    }

    return success ? SDK_SUCCESS : SDK_FAILURE;
}

struct ParsedArgs
{
    size_t H_BINS{0};
    size_t S_BINS{0};
    size_t selectedPlatform{0};
    std::string selectedDevice = "GPU";
    bool outHsv = false;
    bool readBmp = false;
    bool histAsCsv = false;
    bool histGray = false;
    bool inf = false;
    uint64_t maxIter{1};
};

int parseArgument(ParsedArgs& parseArgs, int argc, char * argv[])
{
    struct option long_options[] = {
        {"H_BINS", required_argument, nullptr, 'H'},
        {"S_BINS", required_argument, nullptr, 'S'},
        {"device", required_argument, nullptr, 'D'},
        {"platform", required_argument, nullptr, 'P'},
        {"iteration", required_argument, nullptr, 'i'},
        {"HSV", no_argument, nullptr, 'r'},
        {"BMP", no_argument, nullptr, 'B'},
        {"HistGRAY", no_argument, nullptr, 'G'},
        {"INF", no_argument, nullptr, 'I'},
        {"HistCSV", no_argument, nullptr, 'C'},
        {nullptr, 0, nullptr, 0}         
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "H:S:D:P:i:rBGIC", long_options, nullptr)) != -1) {
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
            case 'i':
                parseArgs.maxIter = std::stoi(optarg);
                parseArgs.inf = true;
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
            case 'G':
                parseArgs.histGray = true;
                break;
            case 'I':
                parseArgs.inf = true;
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
                            parsedArgs.selectedPlatform, parsedArgs.selectedDevice, parsedArgs.histAsCsv, 
                            parsedArgs.histGray, parsedArgs.inf, parsedArgs.maxIter};
                            
    uint64_t iteration{0};
    while (true)
    {
        if (clHistogram.setup() != SDK_SUCCESS) return SDK_FAILURE;
        if (clHistogram.run() != SDK_SUCCESS) return SDK_FAILURE;

        iteration ++;
        if (iteration == parsedArgs.maxIter && ((parsedArgs.maxIter == 1 && !parsedArgs.inf) || parsedArgs.maxIter != 1)) break;

        if(parsedArgs.inf and parsedArgs.maxIter == 1)
        {
            if(iteration == 1)
            {
                readKeyboard::setupKeyboard();
            }
            char key = readKeyboard::pollKey();
            if (key == 27 or key == 'q')
            {
                readKeyboard::cleanupKeyboard();
                break;
            }
        }
    }
    
    return SDK_SUCCESS;
}
