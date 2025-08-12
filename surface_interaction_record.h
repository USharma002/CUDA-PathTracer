#ifndef SURFACE_INTERACTION_RECORDH
#define SURFACE_INTERACTION_RECORDH

#include "vec3.h"
#include "ray.h"

class surface_interaction_record
{
public:
    __host__ __device__ surface_interaction_record() : hit(false), Le(vec3(0, 0, 0)) {}

    bool hit; // did we hit a surface?
    
    vec3 n; // Shading Normal 
    vec3 p; // position of intersection
    vec3 bsdf; // bsdf of intersected surface
    vec3 Le; // Emitted Light

    float t; // distance
};

#endif
