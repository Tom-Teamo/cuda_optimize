# histogram
## 思路
1. 读取global memory： `LDS.128`
2. **直接读取到寄存器，共享内存是block内的histogram结果！！！**
3. **使用原子操作**，因为所有线程都有可能更改shared memory中的结果，所以必须使用`atomicAdd`来保证结果的正确性
3. 写回global memory：也需要使用到原子操作！