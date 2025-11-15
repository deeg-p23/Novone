#pragma once

#include <cstdint>
#include "gdt/math/vec.h"

struct LaunchParams
{
    int frameID { 0 };
    uint32_t *colorBuffer;
    gdt::vec2i fbSize; // frame buffer size
};
