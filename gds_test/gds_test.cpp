#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cufile.h>

#define checkCuda(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        exit(code);
    }
}

int main() {
    const char *test_file = "/mnt/nvme/test_data/gds_sample.bin"; // 确保路径在支持 GDS 的挂载点
    size_t buffer_size = 1024 * 1024; // 1MB

    // 1. 打开文件，必须包含 O_DIRECT
    int fd = open(test_file, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        perror("File open failed (O_DIRECT)");
        return 1;
    }

    // 2. 注册文件句柄到 cuFile
    CUfileHandle_t cfHandle;
    CUfileDescr_t cfDescr = {};
    cfDescr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    cfDescr.handle.fd = fd;

    CUfileError_t status = cuFileHandleRegister(&cfHandle, &cfDescr);
    if (status.err != CU_FILE_SUCCESS) {
        std::cerr << "cuFileHandleRegister failed" << std::endl;
        close(fd);
        return 1;
    }

    // 3. 分配 GPU 显存
    void *devPtr;
    checkCuda(cudaMalloc(&devPtr, buffer_size));

    // 4. 执行 GDS 直接读取
    // 偏移量 0，文件偏移 0
    ssize_t ret = cuFileRead(cfHandle, devPtr, buffer_size, 0, 0);
    if (ret < 0) {
        std::cerr << "cuFileRead failed with error code: " << ret << std::endl;
    } else {
        std::cout << "Successfully read " << ret << " bytes via GPUDirect Storage!" << std::endl;
    }

    // 清理
    cuFileHandleDeregister(cfHandle);
    cudaFree(devPtr);
    close(fd);
    return 0;
}

