#ifndef SURFACE_INTERACTION_RECORDH
#define SURFACE_INTERACTION_RECORDH

#include "vec3.h"
#include "ray.h"

class surface_interaction_record
{
public:
    __host__ __device__ surface_interaction_record() : hit(false), Le(vec3(0, 0, 0)) {}

    bool hit;
    
    vec3 n;
    vec3 p;
    vec3 bsdf;
    vec3 Le;

    float t;
};

#endif
