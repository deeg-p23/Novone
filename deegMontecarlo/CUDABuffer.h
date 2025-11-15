#pragma once

#include "optix9.h"

#include <vector>
#include <assert.h>

struct CUDABuffer
{
    // get device pointer in memory
    CUdeviceptr d_pointer() { return (CUdeviceptr)d_ptr; }
    
    // allocate mem to buffer
    void alloc(size_t size)
    {
        assert(d_ptr == nullptr);   // must be freed before reallocating
        this->sizeInBytes = size;
        CUDA_CHECK(cudaMalloc((void**)&d_ptr, sizeInBytes)); // directive checking for successes in cuda mem calls
    }

    // deallocate mem to buffer and nullify dp
    void free()
    {
        CUDA_CHECK(cudaFree(d_ptr));
        d_ptr = nullptr;
        this->sizeInBytes = 0;
    }

    // free old mem and reallocate new mem to buffer
    void resize(size_t size)
    {
        if (d_ptr) free();
        alloc(size);
    }
    
    // copy data from host (t) to device (buffer)
    template<typename T> void upload(const T *t, size_t count)
    {
        assert(d_ptr != nullptr);
        assert(this->sizeInBytes == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy(d_ptr, (void *)t, count * sizeof(T), cudaMemcpyHostToDevice));
    }

    // copy data from device (buffer) to host (t)
    template<typename T> void download(const T *t, size_t count)
    {
        assert(d_ptr != nullptr);
        assert(this->sizeInBytes == count * sizeof(T));
        CUDA_CHECK(cudaMemcpy((void *) t, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
    }

    // allocate mem and copy vt's data to buffer
    template<typename T> void alloc_and_upload(const std::vector<T> &vt)
    {
        alloc(vt.size() * sizeof(T));
        upload((const T*)vt.data(), vt.size());
    }
    
    size_t sizeInBytes { 0 };
    void *d_ptr { nullptr };
};
