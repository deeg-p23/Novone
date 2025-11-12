#pragma once

#include <cuda_runtime.h> // api for getting cuda device
#include <optix_stubs.h> // for optixInit()
#include <optix_function_table_definition.h> // links g_optixFunctionTable

#include "sutil/Exception.h" // OPTIX_CHECK() : i think it handles host-side exceptions when initializing optix?