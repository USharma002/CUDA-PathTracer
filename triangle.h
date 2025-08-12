#ifndef TRIANGLEH
#define TRIANGLEH

#include "vec3.h"
#include "ray.h"

class triangle
{
public:
    __host__ __device__ triangle() {}

    __host__ __device__ triangle(const vec3& _v0, const vec3& _v1, const vec3& _v2)
        : v0(_v0), v1(_v1), v2(_v2) {}

    // Return intersection point if hit, else return vec3(-1, -1, -1)
    __device__ bool intersect(const ray& r) const
    {
        const float EPSILON = 1e-8f;

        vec3 edge1 = v1 - v0;
        vec3 edge2 = v2 - v0;

        vec3 h = cross(r.direction(), edge2);
        float a = dot(edge1, h);

        if (fabs(a) < EPSILON)
            return false;  // No hit, ray parallel to triangle.

        float f = 1.0f / a;
        vec3 s = r.origin() - v0;
        float u = f * dot(s, h);

        if (u < 0.0f || u > 1.0f)
            return false;  // Outside triangle.

        vec3 q = cross(s, edge1);
        float v = f * dot(r.direction(), q);

        if (v < 0.0f || u + v > 1.0f)
            return false;  // Outside triangle.

        float t = f * dot(edge2, q);

        if (t > EPSILON)
        {
            // Compute intersection point
            return true;        }

        return false;  // No hit (line intersects behind ray origin).
    }

    vec3 v0;
    vec3 v1;
    vec3 v2;
};

#endif
