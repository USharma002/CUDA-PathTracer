#ifndef RAYH
#define RAYH
#include "vector.h"

class Ray
{
    public:
        __host__ __device__  Ray() {}
        __host__ __device__  Ray(const Vector3f& origin, const Vector3f& direction){
            o = origin; // origin 
            d = unit_vector(direction); // direction
        }
        __host__ __device__ Vector3f origin() const { 
            return o; // origin of the ray
        }
        __host__ __device__ Vector3f direction() const { 
            return d; // direction of the ray
        }
        __host__ __device__ Vector3f at(float t) const {
            return o + t*d; // point at distance t along the ray
        }
        __host__ __device__ Vector3f normalize() const { 
            return unit_vector(d); 
        }

        Vector3f o; // Origin of the ray
        Vector3f d; // Direction of the ray
};

#endif