#ifndef SURFACE_INTERACTION_RECORDH
#define SURFACE_INTERACTION_RECORDH

#include "vec3.h"
#include "ray.h"

class surface_interaction_record
{
public:
    __host__ __device__ surface_interaction_record() : hit(false){}

    bool hit;
    
    vec3 normal;
    vec3 position;
    vec3 bsdf;

    float t;
};

#endif
