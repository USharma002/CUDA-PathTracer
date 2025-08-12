#ifndef HITABLELISTH
#define HITABLELISTH

#include "triangle.h"
#include "surface_interaction_record.h"

class scene  {
    public:
        __host__ __device__ scene() {}
        __host__ __device__ scene(triangle *l, int n) {list = l; list_size = n;}
        __device__ bool intersect(const ray& r, float t_min, float t_max, surface_interaction_record &si) const;
        triangle *list;
        int list_size;
};

__device__ bool scene::intersect(const ray& r, float t_min, float t_max, surface_interaction_record &si) const {
        bool hit_anything = false;
        surface_interaction_record temp;
        si.t = t_max;

        for (int i = 0; i < list_size; i++) {
            if (list[i].intersect(r, t_min, t_max, temp)) {
                if (temp.t < si.t) si = temp; // store the closest valid hit
                hit_anything = true;
            }
        }
        return hit_anything;
}

#endif