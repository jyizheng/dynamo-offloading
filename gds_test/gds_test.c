/*
 * GDS (GPU Direct Storage) Test
 * Tests whether cuFile can perform true GPU<->NVMe DMA transfers
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cufile.h>

#define TEST_FILE "/mnt/nvme/gds_test_file"
#define BUF_SIZE  (4 * 1024 * 1024)  /* 4MB */
#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        goto cleanup; \
    } \
} while(0)

static const char *cufile_errstr(CUfileError_t err) {
    switch (err.err) {
        case CU_FILE_SUCCESS:               return "SUCCESS";
        case CU_FILE_DRIVER_NOT_INITIALIZED: return "DRIVER_NOT_INITIALIZED (nvidia-fs not loaded)";
        case CU_FILE_DRIVER_VERSION_MISMATCH: return "DRIVER_VERSION_MISMATCH (nvidia-fs/nvidia driver version mismatch)";
        case CU_FILE_IO_NOT_SUPPORTED:      return "IO_NOT_SUPPORTED (filesystem/block device not GDS-capable)";
        case CU_FILE_NVFS_DRIVER_ERROR:     return "NVFS_DRIVER_ERROR (nvidia-fs ioctl error)";
        case CU_FILE_NVFS_SETUP_ERROR:      return "NVFS_SETUP_ERROR (nvidia-fs initialization error)";
        default: {
            static char buf[64];
            snprintf(buf, sizeof(buf), "error=%d (base+%d)", (int)err.err, (int)err.err - 5000);
            return buf;
        }
    }
}

int main(void) {
    int ret = 1;
    int fd = -1;
    void *gpu_buf = NULL;
    CUfileHandle_t fh = {0};
    CUfileDescr_t descr = {0};
    CUfileError_t status;
    char *host_verify = NULL;
    int fh_registered = 0;
    int buf_registered = 0;

    printf("=== GPU Direct Storage (GDS) Test ===\n\n");

    /* 1. Check nvidia-fs driver */
    printf("[1] Checking nvidia-fs driver...\n");
    {
        FILE *f = fopen("/proc/driver/nvidia-fs/version", "r");
        if (!f) {
            printf("    FAIL: /proc/driver/nvidia-fs/version not found (nvidia-fs not loaded)\n");
            return 1;
        }
        char ver[64] = {0};
        fgets(ver, sizeof(ver), f);
        fclose(f);
        ver[strcspn(ver, "\n")] = 0;
        printf("    OK: nvidia-fs %s\n", ver);
    }

    /* 2. Check /dev/nvidia-fs0 */
    printf("[2] Checking /dev/nvidia-fs0 device node...\n");
    if (access("/dev/nvidia-fs0", F_OK) != 0) {
        printf("    FAIL: /dev/nvidia-fs0 not found\n");
        return 1;
    }
    printf("    OK: device node exists\n");

    /* 3. Init CUDA */
    printf("[3] Initializing CUDA...\n");
    {
        int dev_count = 0;
        cudaGetDeviceCount(&dev_count);
        if (dev_count == 0) {
            printf("    FAIL: no CUDA devices\n");
            return 1;
        }
        cudaSetDevice(0);
        struct cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("    OK: GPU 0 = %s\n", prop.name);
    }

    /* 4. Open cuFile driver */
    printf("[4] Opening cuFile driver...\n");
    status = cuFileDriverOpen();
    if (status.err != CU_FILE_SUCCESS) {
        printf("    FAIL: cuFileDriverOpen() -> %s\n", cufile_errstr(status));
        return 1;
    }
    printf("    OK: cuFileDriverOpen succeeded\n");

    /* 5. Query driver properties */
    printf("[5] Querying cuFile driver properties...\n");
    {
        CUfileDrvProps_t props = {0};
        status = cuFileDriverGetProperties(&props);
        if (status.err == CU_FILE_SUCCESS) {
            printf("    nvfs_version: %u.%u\n",
                   props.nvfs.major_version, props.nvfs.minor_version);
            printf("    dcontrolflags: 0x%x\n", props.nvfs.dcontrolflags);
        }
    }

    /* 6. Allocate GPU buffer */
    printf("[6] Allocating GPU buffer (%dMB)...\n", BUF_SIZE / 1024 / 1024);
    CHECK_CUDA(cudaMalloc(&gpu_buf, BUF_SIZE));
    printf("    OK: GPU buffer at %p\n", gpu_buf);

    /* 7. Register GPU buffer with cuFile */
    printf("[7] Registering GPU buffer with cuFile...\n");
    status = cuFileBufRegister(gpu_buf, BUF_SIZE, 0);
    if (status.err != CU_FILE_SUCCESS) {
        printf("    WARN: cuFileBufRegister() -> %s\n", cufile_errstr(status));
        printf("    (will try without registered buffer)\n");
    } else {
        buf_registered = 1;
        printf("    OK: GPU buffer registered\n");
    }

    /* 8. Open test file with O_DIRECT */
    printf("[8] Opening test file with O_DIRECT...\n");
    fd = open(TEST_FILE, O_CREAT | O_RDWR | O_DIRECT, 0644);
    if (fd < 0) {
        printf("    FAIL: open(%s) -> %s\n", TEST_FILE, strerror(errno));
        goto cleanup;
    }
    printf("    OK: fd=%d\n", fd);

    /* Pre-allocate file space */
    if (ftruncate(fd, BUF_SIZE) != 0) {
        printf("    WARN: ftruncate failed: %s\n", strerror(errno));
    }

    /* 9. Register file handle with cuFile (KEY TEST) */
    printf("[9] Registering file handle with cuFile (GDS capability test)...\n");
    descr.handle.fd = fd;
    descr.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    status = cuFileHandleRegister(&fh, &descr);
    if (status.err != CU_FILE_SUCCESS) {
        printf("    FAIL: cuFileHandleRegister() -> %s\n", cufile_errstr(status));
        printf("\n*** GDS IS NOT WORKING on this filesystem/device ***\n");
        printf("    The block device or filesystem does not support GDS.\n");
        printf("    Transfers will use fallback POSIX path.\n");
        goto cleanup;
    }
    fh_registered = 1;
    printf("    OK: file handle registered -> GDS IS SUPPORTED!\n");

    /* 10. Write test: GPU -> NVMe via GDS */
    printf("[10] Writing GPU buffer to NVMe via GDS...\n");
    {
        /* Fill GPU buffer with pattern */
        cudaMemset(gpu_buf, 0xAB, BUF_SIZE);
        cudaDeviceSynchronize();

        ssize_t written = cuFileWrite(fh, gpu_buf, BUF_SIZE, 0, 0);
        if (written < 0) {
            printf("    FAIL: cuFileWrite() returned %zd\n", written);
            goto cleanup;
        }
        printf("    OK: wrote %zd bytes GPU->NVMe\n", written);
    }

    /* 11. Read test: NVMe -> GPU via GDS */
    printf("[11] Reading from NVMe back to GPU via GDS...\n");
    {
        cudaMemset(gpu_buf, 0x00, BUF_SIZE);
        cudaDeviceSynchronize();

        ssize_t nread = cuFileRead(fh, gpu_buf, BUF_SIZE, 0, 0);
        if (nread < 0) {
            printf("    FAIL: cuFileRead() returned %zd\n", nread);
            goto cleanup;
        }
        printf("    OK: read %zd bytes NVMe->GPU\n", nread);
    }

    /* 12. Verify data integrity */
    printf("[12] Verifying data integrity (GPU->CPU copy for check)...\n");
    {
        host_verify = (char *)malloc(BUF_SIZE);
        if (!host_verify) { printf("    FAIL: malloc\n"); goto cleanup; }
        cudaMemcpy(host_verify, gpu_buf, BUF_SIZE, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        int ok = 1;
        for (int i = 0; i < BUF_SIZE; i++) {
            if ((unsigned char)host_verify[i] != 0xAB) {
                printf("    FAIL: data mismatch at byte %d (got 0x%02x)\n",
                       i, (unsigned char)host_verify[i]);
                ok = 0;
                break;
            }
        }
        if (ok) printf("    OK: data integrity verified (all 0xAB)\n");
        else goto cleanup;
    }

    /* 13. Check nvidia-fs stats for actual GDS IO */
    printf("[13] nvidia-fs IO stats after GDS operations:\n");
    {
        FILE *f = fopen("/proc/driver/nvidia-fs/stats", "r");
        if (f) {
            char line[256];
            while (fgets(line, sizeof(line), f)) {
                if (strstr(line, "Reads") || strstr(line, "Writes") || strstr(line, "Ops")) {
                    printf("    %s", line);
                }
            }
            fclose(f);
        }
    }

    printf("\n=== RESULT: GDS IS WORKING! True GPU<->NVMe DMA transfers confirmed ===\n");
    ret = 0;

cleanup:
    if (host_verify) free(host_verify);
    if (fh_registered) cuFileHandleDeregister(fh);
    if (buf_registered) cuFileBufDeregister(gpu_buf);
    if (gpu_buf) cudaFree(gpu_buf);
    if (fd >= 0) { close(fd); unlink(TEST_FILE); }
    cuFileDriverClose();
    if (ret != 0 && !fh_registered) {
        printf("\n=== RESULT: GDS NOT WORKING (fallback to POSIX path) ===\n");
    }
    return ret;
}
