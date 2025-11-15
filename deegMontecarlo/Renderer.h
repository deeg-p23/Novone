#pragma once

#include "CUDABuffer.h"
#include "LaunchParams.h"

class Renderer
{
    public:
        Renderer();
        void render();
        void resize(gdt::vec2i &newSize);
        void downloadPixels(uint32_t h_pixels[]);
    
    protected:
        // optix init helper functions
        void initOptix();
        void createContext();
        void createModule(); // module contains all programs

        // program init helper functions
        void createRaygenPrograms();
        void createMissPrograms();
        void createHitgroupPrograms();
        void createPipeline();
        void buildSBT();    // shader binding table (OWL can otherwise handle this)

        /*! @{ CUDA device context and stream that optix pipeline will run
            on, as well as device properties for this device */
        CUcontext          cudaContext;
        CUstream           stream;
        cudaDeviceProp     deviceProps;
        /*! @} */

        //! the optix context that our pipeline will run in.
        OptixDeviceContext optixContext;

        /*! @{ the pipeline we're building */
        OptixPipeline               pipeline;
        OptixPipelineCompileOptions pipelineCompileOptions = {};
        OptixPipelineLinkOptions    pipelineLinkOptions    = {};
        /*! @} */

        /*! @{ the module that contains out device programs */
        OptixModule                 module;
        OptixModuleCompileOptions   moduleCompileOptions = {};
        /* @} */

        /*! vector of all our program(group)s, and the SBT built around
            them */
        std::vector<OptixProgramGroup> raygenPGs;
        CUDABuffer raygenRecordsBuffer;
        std::vector<OptixProgramGroup> missPGs;
        CUDABuffer missRecordsBuffer;
        std::vector<OptixProgramGroup> hitgroupPGs;
        CUDABuffer hitgroupRecordsBuffer;
        OptixShaderBindingTable sbt = {};

        /*! @{ our launch parameters, on the host, and the buffer to store
            them on the device */
        LaunchParams launchParams;
        CUDABuffer   launchParamsBuffer;
        /*! @} */

        CUDABuffer colorBuffer;
};
