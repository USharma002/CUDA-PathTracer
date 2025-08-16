#ifndef SURFACE_INTERACTION_RECORDH
#define SURFACE_INTERACTION_RECORDH

#include "vec3.h"
#include "ray.h"

class SurfaceInteractionRecord
{
public:
    __host__ __device__ SurfaceInteractionRecord() : hit(false), Le(Vector3f(0, 0, 0)) {}

    bool hit; // did we hit a surface?
    
    Vector3f n; // Shading Normal 
    Vector3f p; // position of intersection
    Vector3f bsdf; // bsdf of intersected surface
    Vector3f Le; // Emitted Light

    float t; // distance
};

#endif
