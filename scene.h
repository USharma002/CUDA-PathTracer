#ifndef HITABLELISTH
#define HITABLELISTH

#include "triangle.h"
#include "surface_interaction_record.h"

class Scene  {
    public:
        __host__ __device__ Scene() {}
        __host__ __device__ Scene(Triangle *l, int n) {triangles = l; list_size = n;}
        __device__ bool intersect(const Ray& r, float t_min, float t_max, SurfaceInteractionRecord &si) const;
        Triangle *triangles;
        int list_size;
};

__device__ bool Scene::intersect(const Ray& r, float t_min, float t_max, SurfaceInteractionRecord &si) const {
        bool hit_anything = false;
        SurfaceInteractionRecord temp;
        si.t = t_max;

        for (int i = 0; i < list_size; i++) {
            if (triangles[i].intersect(r, t_min, t_max, temp)) {
                if (temp.t < si.t) si = temp; // store the closest valid hit
                hit_anything = true;
            }
        }
        return hit_anything;
}

#endif