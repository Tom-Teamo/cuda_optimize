#include <cuda_runtime.h>
/*
    在这个例子中，使用了grid-stride loops：
        传统的读取数据的操作都是 one thread per data element，称这种kernel为monolithic kernel
        在gird-stride loops中，循环的增长是blockDim.x*gridDim.x，所以如果有1280个线程在grid中，那么线程0将读取0 1280 ...
    声称可以达到全局内存的合并访问

    向量化访存：
        即将原始的数据类型转换为 cuda中定义的 数据类型 例如 int2 int4，它们分别可以表示两个int 4个int
        在#include <vector_types.h>中可以找到所有支持的 向量化数据类型

        通过reinterpret_cast<int2*>(d_out)，首先将数据的内存转化为int2格式的，然后再赋值，这样每次都是执行两次正常的int赋值了


        1. 假设用的是正常的float，那么读取4个float的话，需要发射4条LD.E指令。而使用float4的话，只需要发射一条LD.E.128指令
            Nvidia的GPU中，由于SIMT架构是通过切换warp来掩盖访存的延时，
            所以并不代表着4个cycle发射4条指令就比1个cycle发射1条指令慢4倍，大部分的时间其实都是在等待访存单元把数据拿回来，
            而真正访存的时间，不管是去L1还是L2拿数，cache line都是128Byte。float和float4都是一样。
            对于一个warp而言，如果32个线程想要去拿128个数，不管float还是float4，都得变成4次对cache line的读取，当然，
            如果仅仅是拷贝，这个cache line还不一定命中。如果到了global mem中去读，从硬件的角度而言，
            访存端口也是一样，并不会因为float4就能够获得更多的端口读数。所以结论是，float4会更快，但是快得不多。

        2. 对于指令cache而言，所需要的指令更少了，那么icache不命中的概率就会减少很多。
        我们假设一个场景，对于一段核心代码，volta架构有12KB的L0 指令cache，一条指令需要128bit，
        那么最多可以容纳768条SASS指令，对于sgemm中的核心循环，假设取12行12列，组成12x12=144条FFMA指令，
        循环展开6次，144x6=576条指令，一共有(12+12)*6 = 144个数需要load，如果用float则需要144条指令，
        那么计算和访存一共有720条指令，再加一些其他的指令，很容易导致指令cache放不下，性能有所损失。
        如果用float4的话，则需要144/4 = 36条指令，总共612条指令，指令cache肯定能放得下。

        当然，转成float4也会产生一些负面的影响，首先是所采用的寄存器更多了，寄存器资源被占用多了之后，
        SM中能够并发的warp数量会有所减少。此外，如果本身程序的并行粒度就不太够，使用float4的话，
        所使用的block数量减少，warp数量减少，性能也会有一定的影响。

*/

__global__ void device_copy_scalar_kernel(int* d_in, int* d_out, int N) { 
  int idx = blockIdx.x * blockDim.x + threadIdx.x; 
  for (int i = idx; i < N; i += blockDim.x * gridDim.x) { 
    d_out[i] = d_in[i]; 
  } 
} 

void device_copy_scalar(int* d_in, int* d_out, int N) 
{ 
  int threads = 128; 
  int blocks = min((N + threads-1) / threads, 1280000);  
  device_copy_scalar_kernel<<<blocks, threads>>>(d_in, d_out, N); 
}


__global__ void device_copy_vector2_kernel(int* d_in, int* d_out, int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = idx; i < N/2; i += blockDim.x * gridDim.x) {
    reinterpret_cast<int2*>(d_out)[i] = reinterpret_cast<int2*>(d_in)[i];
  }

  // in only one thread, process final element (if there is one)
  if (idx==N/2 && N%2==1)
    d_out[N-1] = d_in[N-1];
}

void device_copy_vector2(int* d_in, int* d_out, int n) {
  int threads = 128; 
  int blocks = min((n/2 + threads-1) / threads, 1280000); 

  device_copy_vector2_kernel<<<blocks, threads>>>(d_in, d_out, n);
}

int main() {

    return 0;
}