#ifndef TRIANGLEH
#define TRIANGLEH

#include "vec3.h"
#include "ray.h"
#include "surface_interaction_record.h"

class triangle
{
public:
    __host__ __device__ triangle() {}

    __host__ __device__ triangle(const vec3& _v0, const vec3& _v1, const vec3& _v2, const vec3& _bsdf = vec3(0.3f, 0.2f, 0.3f))
        : v0(_v0), v1(_v1), v2(_v2), bsdf(_bsdf){}

    __host__ __device__ triangle(const vec3& _v0, const vec3& _v1, const vec3& _v2, const vec3& _bsdf, const vec3& _normal)
    : v0(_v0), v1(_v1), v2(_v2), bsdf(_bsdf), normal(_normal) {}

    __device__ bool intersect(const ray& r, float t_min, float t_max, surface_interaction_record &si) const
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

        if (t > EPSILON && t >= t_min && t <= t_max)
        {
            si.t = t;
            si.hit = true;
            si.n = normal;
            si.bsdf = bsdf;
            si.Le = Le;
            si.p = r.origin() + t * r.direction();

            // Compute intersection point
            return true;        
        }

        return false;  // No hit (line intersects behind ray origin).
    }

    vec3 v0;
    vec3 v1;
    vec3 v2;

    vec3 bsdf;
    vec3 normal;
    vec3 Le;
};

#endif
