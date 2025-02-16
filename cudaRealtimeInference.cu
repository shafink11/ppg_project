#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>

#define CUDA_CHECK(status) \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    }

#define CUDNN_CHECK(status) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        std::cerr << "cuDNN Error: " << cudnnGetErrorString(status) << std::endl; \
        exit(EXIT_FAILURE); \
    }

namespace fs = std::filesystem;


std::vector<float> loadWaveform(const std::string& filepath) {
    std::vector<float> waveform;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return waveform;
    }
    float value;
    while (file >> value) {
        waveform.push_back(value);
    }
    return waveform;
}

std::vector<std::vector<float>> loadAllWaveforms(const std::string& directoryPath) {
    std::vector<std::vector<float>> allWaveforms;
    for (const auto& entry : fs::directory_iterator(directoryPath)) {
        if (entry.is_regular_file()) {
            std::string path = entry.path().string();
            std::vector<float> waveform = loadWaveform(path);
            if (!waveform.empty()) {
                std::cout << "Loaded waveform from " << path 
                          << " with " << waveform.size() << " samples." << std::endl;
                allWaveforms.push_back(waveform);
            }
        }
    }
    return allWaveforms;
}

__global__ void conv1d_kernel(const float* __restrict__ input,
                                const float* __restrict__ kernel,
                                float* __restrict__ output,
                                int input_size,
                                int kernel_size)
{
    extern __shared__ float shared_input[];
    int tid = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x;
    int out_idx = blockStart + tid;
    for (int i = tid; i < blockDim.x + kernel_size - 1; i += blockDim.x) {
        int input_idx = blockStart + i;
        shared_input[i] = (input_idx < input_size) ? input[input_idx] : 0.0f;
    }
    __syncthreads();
    if (out_idx < (input_size - kernel_size + 1)) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < kernel_size; ++k) {
            sum += shared_input[tid + k] * kernel[k];
        }
        output[out_idx] = sum;
    }
}

void runCUDNNConv1D() {
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    int batch = 1, channels = 1, height = 1, width = 1024;
    int kernel_width = 5;

    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(input_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           batch, channels, height, width));

    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filter_desc,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,
                                           1, 1, height, kernel_width));

    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc));
    int pad_h = 0, pad_w = 0;
    int stride_h = 1, stride_w = 1;
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(conv_desc,
                                                pad_h, pad_w,
                                                stride_h, stride_w,
                                                1, 1,  
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    int out_n, out_c, out_h, out_w;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(conv_desc,
                                                      input_desc,
                                                      filter_desc,
                                                      &out_n, &out_c, &out_h, &out_w));

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(output_desc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           out_n, out_c, out_h, out_w));

    float *d_input, *d_filter, *d_output;
    size_t input_bytes = batch * channels * height * width * sizeof(float);
    size_t filter_bytes = 1 * 1 * height * kernel_width * sizeof(float);
    size_t output_bytes = out_n * out_c * out_h * out_w * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_input, input_bytes));
    CUDA_CHECK(cudaMalloc(&d_filter, filter_bytes));
    CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

    cudnnConvolutionFwdAlgo_t algo;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                    input_desc,
                                                    filter_desc,
                                                    conv_desc,
                                                    output_desc,
                                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                    0,
                                                    &algo));

    size_t workspace_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                        input_desc,
                                                        filter_desc,
                                                        conv_desc,
                                                        output_desc,
                                                        algo,
                                                        &workspace_bytes));
    void* d_workspace = nullptr;
    CUDA_CHECK(cudaMalloc(&d_workspace, workspace_bytes));

    const float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(cudnn,
                                        &alpha,
                                        input_desc,
                                        d_input,
                                        filter_desc,
                                        d_filter,
                                        conv_desc,
                                        algo,
                                        d_workspace,
                                        workspace_bytes,
                                        &beta,
                                        output_desc,
                                        d_output));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_filter));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_workspace));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(output_desc));
    CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc));
    CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_desc));
    CUDNN_CHECK(cudnnDestroy(cudnn));
}

#include <NvInfer.h>
using namespace nvinfer1;

class TRTLogger : public ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity != Severity::kINFO)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
} gLogger;

ICudaEngine* createTRTInferenceEngine() {
    IBuilder* builder = createInferBuilder(gLogger);
    if (!builder) {
        std::cerr << "Failed to create TensorRT builder!" << std::endl;
        return nullptr;
    }
    INetworkDefinition* network = builder->createNetworkV2(0U);

    ITensor* input = network->addInput("input", DataType::kFLOAT, Dims3{1, 1, 1024});
    if (!input) {
        std::cerr << "Failed to create input tensor!" << std::endl;
        return nullptr;
    }

    Weights {DataType::kFLOAT, std::kload(weights), 0}; 
    IConvolutionLayer* conv = network->addConvolutionNd(*input, 16, DimsHW{1, 5}, emptyWeights, emptyWeights);
    conv->setStrideNd(DimsHW{1, 1});
    conv->setPaddingNd(DimsHW{0, 0});
    if (!conv) {
        std::cerr << "Failed to add convolution layer!" << std::endl;
        return nullptr;
    }

    IActivationLayer* relu = network->addActivation(*conv->getOutput(0), ActivationType::kRELU);
    if (!relu) {
        std::cerr << "Failed to add activation layer!" << std::endl;
        return nullptr;
    }

    network->markOutput(*relu->getOutput(0));

    IBuilderConfig* config = builder->createBuilderConfig();
    config->setMaxWorkspaceSize(1 << 20); 
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);

    network->destroy();
    config->destroy();
    builder->destroy();

    return engine;
}

//------------------------------------------------------------------------------
// Main: Load data and process using CUDA and neural network components
//------------------------------------------------------------------------------
int main() {
    std::string dataDir = "./processed_ppgs";
    std::vector<std::vector<float>> waveforms = loadAllWaveforms(dataDir);
    if (waveforms.empty()) {
        std::cerr << "No waveforms loaded from " << dataDir << std::endl;
        return -1;
    }
    std::vector<float>& ppg = waveforms[0];
    int input_size = ppg.size();
    int kernel_size = 5;
    int output_size = input_size - kernel_size + 1;

    std::vector<float> kernel(kernel_size, 1.0f / kernel_size);

    float* h_input = ppg.data();
    float* h_kernel = kernel.data();
    std::vector<float> h_output(output_size, 0.0f);

    float *d_input, *d_kernel, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_size * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocks = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemBytes = (threadsPerBlock + kernel_size - 1) * sizeof(float);
    conv1d_kernel<<<blocks, threadsPerBlock, sharedMemBytes>>>(d_input, d_kernel, d_output, input_size, kernel_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < 10 && i < output_size; ++i)
        std::cout << h_output[i] << " ";
    std::cout << std::endl;

    runCUDNNConv1D();

    ICudaEngine* engine = createTRTInferenceEngine();
    if (engine) {
        std::cout << "TensorRT engine created successfully!" << std::endl;
        engine->destroy();
    } else {
        std::cerr << "Failed to create TensorRT engine." << std::endl;
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
