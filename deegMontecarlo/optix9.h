#pragma once

#include <cuda_runtime.h> // api for getting cuda device
#include <optix_stubs.h> // for optixInit()
#include <optix_function_table_definition.h> // links g_optixFunctionTable

#include "sutil/Exception.h" // OPTIX_CHECK() : i think it handles host-side exceptions when initializing optix?



// macro definitions for initializing cuda & optix
// LESSON: first time working with the #define directive
//  - macro must be inline, hence the backslashes
//  - parameters do not have type declaration

// cuda check directive without throwing exceptions (not sure what the purpose/use-case is yet)
#define CUDA_CHECK_NOEXCEPT( call )                                                                                    \
{									                                                                                   \
    cuda##call;                                                                                                        \
}

// all other directive checks, OPTIX_CHECK, CUDA_CHECK, and CUDA_SYNC_CHECK defined in sutil/Exception.h
