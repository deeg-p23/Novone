#include "Renderer.h"

// LESSON: uses the keyword extern "C" b/c it is defined elsewhere (devicePrograms.cu)
extern "C" char devicePrograms_ptx[];

// LESSON: __align__ specifier aligns struct data to 16ull (unsigned-long-long)
// SBT record of a raygen program
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
};
// TODO: read on the significance of program records