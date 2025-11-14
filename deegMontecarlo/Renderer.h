#pragma once

#include "gdt/math/vec.h"

class Renderer
{
    public:
        Renderer();
        void render();
        void resize(const gdt::vec2i &newSize);
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

    
};
