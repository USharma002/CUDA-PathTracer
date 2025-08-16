#ifndef TRIANGLEH
#define TRIANGLEH

#include "vec3.h"
#include "ray.h"
#include "surface_interaction_record.h"

class Triangle
{
public:
    __host__ __device__ Triangle() {}

    __host__ __device__ Triangle(const Vector3f& _v0, const Vector3f& _v1, const Vector3f& _v2, const Vector3f& _bsdf = Vector3f(0.3f, 0.2f, 0.3f))
        : v0(_v0), v1(_v1), v2(_v2), bsdf(_bsdf){}

    __host__ __device__ Triangle(const Vector3f& _v0, const Vector3f& _v1, const Vector3f& _v2, const Vector3f& _bsdf, const Vector3f& _normal)
    : v0(_v0), v1(_v1), v2(_v2), bsdf(_bsdf), normal(_normal) {}

    __device__ bool intersect(const Ray& r, float t_min, float t_max, SurfaceInteractionRecord &si) const
    {
        const float EPSILON = 1e-8f;

        Vector3f edge1 = v1 - v0;
        Vector3f edge2 = v2 - v0;

        Vector3f h = cross(r.direction(), edge2);
        float a = dot(edge1, h);

        if (fabs(a) < EPSILON)
            return false;  // No hit, ray parallel to triangle.

        float f = 1.0f / a;
        Vector3f s = r.origin() - v0;
        float u = f * dot(s, h);

        if (u < 0.0f || u > 1.0f)
            return false;  // Outside triangle.

        Vector3f q = cross(s, edge1);
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

    Vector3f v0;
    Vector3f v1;
    Vector3f v2;

    Vector3f bsdf;
    Vector3f normal;
    Vector3f Le;
};

#endif