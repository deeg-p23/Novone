#include "optix9.h"

#include <iostream>
#include <ostream>
#include <stdexcept>

int main( int argc, char* argv[] )
{
    try
    {
        // getting cuda device
        cudaFree(0);
        int numDevices;
        cudaGetDeviceCount(&numDevices);
        if (numDevices == 0) throw std::runtime_error("No devices found");
        
        // initialize optix
        OPTIX_CHECK(optixInit());
    } catch (std::runtime_error& e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
