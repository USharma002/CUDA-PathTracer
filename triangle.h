#ifndef TRIANGLEH
#define TRIANGLEH

#include "vector.h"
#include "ray.h"
#include "surface_interaction_record.h"

#ifndef GRID_SIZE
#define GRID_SIZE 400  // 100x100 default
#endif

#ifndef RADIOSITY_HISTORY
#define RADIOSITY_HISTORY 10  // 100x100 default
#endif

class Triangle {
public:
    __host__ __device__ Triangle() {
        for (int i = 0; i < GRID_SIZE; i++) {
            grid[i] = 0.0f;
            radiosity_grid[i] = Vector3f(0.0f, 0.0f, 0.0f);  // Initialize to black
        }
    }

    __host__ __device__ Triangle(const Vector3f& _v0, const Vector3f& _v1, const Vector3f& _v2,
                                const Vector3f& _bsdf = Vector3f(0.3f, 0.2f, 0.3f))
        : v0(_v0), v1(_v1), v2(_v2), bsdf(_bsdf) {
        Vector3f edge1 = v1 - v0;
        Vector3f edge2 = v2 - v0;
        normal = unit_vector(cross(edge1, edge2));
        area = 0.5f * cross(edge1, edge2).length();
        Le = Vector3f(0.0f, 0.0f, 0.0f);
        radiosity = Vector3f(0.0f, 0.0f, 0.0f);
        unshot_rad = Vector3f(0.0f, 0.0f, 0.0f);
        v0_rad = Vector3f(0.0f, 0.0f, 0.0f);
        v1_rad = Vector3f(0.0f, 0.0f, 0.0f);
        v2_rad = Vector3f(0.0f, 0.0f, 0.0f);
        
        for (int i = 0; i < GRID_SIZE; i++) {
            grid[i] = 0.0f;
            radiosity_grid[i] = Vector3f(0.0f, 0.0f, 0.0f);
        }

        history_index = 0;
        history_count = 0;
        for (int i = 0; i < RADIOSITY_HISTORY; i++) {
            radiosity_history[i] = Vector3f(0.0f, 0.0f, 0.0f);
        }
    }

    // ADD THIS CONSTRUCTOR - 5 argument version
    __host__ __device__ Triangle(const Vector3f& _v0, const Vector3f& _v1, const Vector3f& _v2, 
                                 const Vector3f& _bsdf, const Vector3f& _normal)
        : v0(_v0), v1(_v1), v2(_v2), bsdf(_bsdf), normal(_normal) {
        Vector3f edge1 = v1 - v0;
        Vector3f edge2 = v2 - v0;
        area = 0.5f * cross(edge1, edge2).length();
        
        Le = Vector3f(0.0f, 0.0f, 0.0f);
        radiosity = Vector3f(0.0f, 0.0f, 0.0f);
        unshot_rad = Vector3f(0.0f, 0.0f, 0.0f);
        v0_rad = Vector3f(0.0f, 0.0f, 0.0f);
        v1_rad = Vector3f(0.0f, 0.0f, 0.0f);
        v2_rad = Vector3f(0.0f, 0.0f, 0.0f);
    }
    
    __host__ __device__ bool intersect(const Ray& r, float t_min, float t_max, SurfaceInteractionRecord &si) const {
        const float EPSILON = 1e-8f;
        Vector3f edge1 = v1 - v0;
        Vector3f edge2 = v2 - v0;
        Vector3f h = cross(r.direction(), edge2);
        float a = dot(edge1, h);
        
        if (fabs(a) < EPSILON) return false;
        
        float f = 1.0f / a;
        Vector3f s = r.origin() - v0;
        float u = f * dot(s, h);
        if (u < 0.0f || u > 1.0f) return false;
        
        Vector3f q = cross(s, edge1);
        float v = f * dot(r.direction(), q);
        if (v < 0.0f || u + v > 1.0f) return false;
        
        float t = f * dot(edge2, q);
        if (t > EPSILON && t >= t_min && t <= t_max) {
            si.t = t;
            si.hit = true;
            si.n = normal;
            si.bsdf = bsdf;
            si.Le = Le;
            si.radiosity = radiosity;
            si.p = r.origin() + t * r.direction();
            si.u = u;
            si.v = v;
            return true;
        }
        return false;
    }
    
    Vector3f v0, v1, v2;
    Vector3f bsdf;
    Vector3f normal;
    Vector3f Le;
    Vector3f radiosity;
    Vector3f unshot_rad;
    float area;
    Vector3f v0_rad, v1_rad, v2_rad;
    
    float grid[GRID_SIZE];
    Vector3f radiosity_grid[GRID_SIZE];

    Vector3f radiosity_history[RADIOSITY_HISTORY];
    int history_index; // Current write position (0-9)
    int history_count; // Number of valid entries (0-10)
};

#endif
