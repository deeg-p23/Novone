#pragma once

#include "gdt/math/vec.h"
#include <cstdint>

struct LaunchParams
{
    int frameID { 0 };
    uint32_t *colorBuffer;
    vec2i fbSize; // frame buffer size
};
