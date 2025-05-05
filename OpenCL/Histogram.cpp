#include "Histogram.hpp"


std::vector<unsigned int> histogram(H_BINS * S_BINS);

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

int Histogram::writeOutputImage(std::string outputImageName)
{
    memcpy(pixelData, outputImageData, width * height * pixelSize);
    return inputBitmap.write(outputImageName.c_str()) ? SDK_SUCCESS : SDK_FAILURE;
}

int Histogram::setupCL()
{
    cl_int err;
    cl::Platform::get(&platforms);
    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
    context = cl::Context(CL_DEVICE_TYPE_GPU, cps, NULL, NULL, &err);
    devices = context.getInfo<CL_CONTEXT_DEVICES>();
    commandQueue = cl::CommandQueue(context, devices[0], 0, &err);
    inputImage2D = cl::Image2D(context, CL_MEM_READ_ONLY, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), width, height);
    outputImage2D = cl::Image2D(context, CL_MEM_WRITE_ONLY, cl::ImageFormat(CL_RGBA, CL_UNSIGNED_INT8), width, height);

    SDKFile kernelFile;
    std::string kernelPath = getPath() + "Histogram_Kernels.cl";
    kernelFile.open(kernelPath.c_str());
    cl::Program::Sources source(1, std::make_pair(kernelFile.source().data(), kernelFile.source().size()));
    program = cl::Program(context, source);
    program.build(devices);
    kernel = cl::Kernel(program, "rgb2hsv");
    
    histogramBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, H_BINS * S_BINS * sizeof(cl_uint));

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
    
    commandQueue.enqueueWriteImage(inputImage2D, CL_TRUE, origin, region, 0, 0, inputImageData);
    kernel.setArg(0, inputImage2D);
    kernel.setArg(1, outputImage2D);
    cl::NDRange global(width, height);
    cl::NDRange local(1, 1);
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
    commandQueue.enqueueReadImage(outputImage2D, CL_TRUE, origin, region, 0, 0, outputImageData);


    // Kernel 2: HSV Histogram
    cl::Kernel histKernel(program, "hsvHistogram");
    histKernel.setArg(0, outputImage2D);       // HSV obraz z poprzedniego kroku
    histKernel.setArg(1, histogramBuffer);
    histKernel.setArg(2, width);
    histKernel.setArg(3, height);
    histKernel.setArg(4, H_BINS);
    histKernel.setArg(5, S_BINS);
    commandQueue.enqueueNDRangeKernel(histKernel, cl::NullRange, global, local);

    // Odczytaj histogram
    commandQueue.enqueueReadBuffer(histogramBuffer, CL_TRUE, 0, histogram.size() * sizeof(cl_uint), histogram.data());
    
    return SDK_SUCCESS;
}

int Histogram::setup()
{
    std::string path = getPath() + INPUT_IMAGE;
    return readInputImage(path) == SDK_SUCCESS && setupCL() == SDK_SUCCESS ? SDK_SUCCESS : SDK_FAILURE;
}

int Histogram::run()
{
    return runCLKernels() == SDK_SUCCESS && writeOutputImage(OUTPUT_IMAGE) == SDK_SUCCESS ? SDK_SUCCESS : SDK_FAILURE;
}

int Histogram::cleanup()
{
    free(inputImageData);
    free(outputImageData);
    return SDK_SUCCESS;
}

int main(int argc, char * argv[])
{
    Histogram clHistogram;
    if (clHistogram.setup() != SDK_SUCCESS) return SDK_FAILURE;
    if (clHistogram.run() != SDK_SUCCESS) return SDK_FAILURE;
    if (clHistogram.cleanup() != SDK_SUCCESS) return SDK_FAILURE;
    return SDK_SUCCESS;
}
