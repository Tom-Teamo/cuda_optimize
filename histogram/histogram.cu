#include <iostream>
#include <cuda_runtime.h>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define vectorPtr(x) thrust::raw_pointer_cast(x.data())

void cpu_compute_hist(const thrust::host_vector<uint8_t>& input, 
                        thrust::host_vector<int>& hist) {
    size_t s = input.size();

    for (size_t i = 0; i < s; i++) {
        hist[input[i]] ++;
    }
}

__global__ void gpu_compute_hist(uint8_t* input, int* hist, int input_size) {
    int input_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    
    if (input_idx * 4 >= input_size) return;

    __shared__ int s_mem[256];

    if (tid < 256) {
        s_mem[tid] = 0;
    }

    __syncthreads();

    uchar4 data = reinterpret_cast<uchar4*>(input)[input_idx];
    atomicAdd(&s_mem[data.x], 1);
    atomicAdd(&s_mem[data.y], 1);
    atomicAdd(&s_mem[data.z], 1);
    atomicAdd(&s_mem[data.w], 1);

    __syncthreads();

    if (tid < 256) {
        atomicAdd(&hist[tid], s_mem[tid]);
    }

}


int main() {
    size_t img_w = 2048;
    size_t img_h = 2048;
    size_t img_size = img_h * img_w;

    // malloc the input
    thrust::host_vector<uint8_t> input(img_size, 0);

    // init the input
    for (size_t i = 0; i < img_size; i++) {
        input[i] = rand() % 256;
    }

    // compute the histogram result in CPU
    thrust::host_vector<int> cpu_histogram(256, 0);
    cpu_compute_hist(input, cpu_histogram);

    // init the input in device-end
    thrust::device_vector<uint8_t> input_device(input.begin(), input.end());
    thrust::device_vector<int> gpu_histogram(256, 0);

    // each thread computes 4 element
    size_t block_size = 128 * 8;
    dim3 block(block_size);
    dim3 grid((img_size + block_size * 4 - 1) / (block_size * 4));

    gpu_compute_hist<<<grid, block>>>(vectorPtr(input_device), vectorPtr(gpu_histogram), img_size);

    thrust::host_vector<int> gpu_hist_host(256, 0);
    thrust::copy(gpu_histogram.begin(), gpu_histogram.end(), gpu_hist_host.begin());

    for (size_t i = 0; i < 256; i++) {
        if (cpu_histogram[i] != gpu_hist_host[i]) {
            std::cout << "shit at: " << i << std::endl;
            std::cout << "cpu at: " << i << " is: " << cpu_histogram[i] << std::endl;
            std::cout << "gpu at: " << i << " is: " << gpu_histogram[i] << std::endl;
        }
    }
    
    return 0;
}