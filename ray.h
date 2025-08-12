#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray
{
    public:
        __host__ __device__  ray() {}
        __host__ __device__  ray(const vec3& a, const vec3& b) { o = a; d = b; }
        __device__ vec3 origin() const       { return o; }
        __device__ vec3 direction() const    { return d; }
        __device__ vec3 point_at_parameter(float t) const { return o + t*d; }
        __device__ vec3 normalize() const       { return unit_vector(d); }

        vec3 o;
        vec3 d;
};

#endif