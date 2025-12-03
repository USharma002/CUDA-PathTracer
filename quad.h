#ifndef QUAD_H
#define QUAD_H

#include "vector.h"
#include "ray.h"
#include "surface_interaction_record.h"
#include "render_config.h"

#ifndef RADIOSITY_HISTORY
#define RADIOSITY_HISTORY 10
#endif

class Quad {
public:
    __host__ __device__ Quad() {
        for (int i = 0; i < GRID_SIZE; i++) {
            grid[i] = 0.0f;
            radiosity_grid[i] = Vector3f(0.0f, 0.0f, 0.0f);
        }
    }

    __host__ __device__ Quad(const Vector3f& _v00, const Vector3f& _v10,
                             const Vector3f& _v11, const Vector3f& _v01,
                             const Vector3f& _bsdf = Vector3f(0.8f, 0.8f, 0.8f))
        : v00(_v00), v10(_v10), v11(_v11), v01(_v01), bsdf(_bsdf) {
        
        Vector3f edge1 = v10 - v00;
        Vector3f edge2 = v01 - v00;
        normal = unit_vector(cross(edge1, edge2));
        
        area = 0.5f * (cross(edge1, edge2).length() + cross(v11 - v10, v11 - v01).length());
        
        Le = Vector3f(0.0f, 0.0f, 0.0f);
        radiosity = Vector3f(0.0f, 0.0f, 0.0f);
        unshot_rad = Vector3f(0.0f, 0.0f, 0.0f);
        
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

    __host__ __device__ bool intersect(const Ray& r, float t_min, float t_max,
                                      SurfaceInteractionRecord& si) const {
        const float EPSILON = 1e-8f;
        bool hit = false;
        float closest_t = t_max;
        float best_u = 0.0f;
        float best_v = 0.0f;
        
        // Test first triangle (v00, v10, v11) - ADD BRACES!
        {
            Vector3f edge1 = v10 - v00;
            Vector3f edge2 = v11 - v00;
            Vector3f h = cross(r.direction(), edge2);
            float a = dot(edge1, h);
            
            if (fabs(a) > EPSILON) {
                float f = 1.0f / a;
                Vector3f s = r.origin() - v00;
                float u = f * dot(s, h);
                
                if (u >= 0.0f && u <= 1.0f) {
                    Vector3f q = cross(s, edge1);
                    float v = f * dot(r.direction(), q);
                    
                    if (v >= 0.0f && u + v <= 1.0f) {
                        float t = f * dot(edge2, q);
                        
                        if (t > EPSILON && t >= t_min && t < closest_t) {
                            closest_t = t;
                            hit = true;
                            best_u = u;  // Store temporarily
                            best_v = v;
                        }
                    }
                }
            }
        } // CRITICAL: Close scope here!
        
        // Test second triangle (v00, v11, v01) - ADD BRACES!
        {
            Vector3f edge1 = v11 - v00;
            Vector3f edge2 = v01 - v00;
            Vector3f h = cross(r.direction(), edge2);
            float a = dot(edge1, h);
            
            if (fabs(a) > EPSILON) {
                float f = 1.0f / a;
                Vector3f s = r.origin() - v00;
                float u = f * dot(s, h);
                
                if (u >= 0.0f && u <= 1.0f) {
                    Vector3f q = cross(s, edge1);
                    float v = f * dot(r.direction(), q);
                    
                    if (v >= 0.0f && u + v <= 1.0f) {
                        float t = f * dot(edge2, q);
                        
                        if (t > EPSILON && t >= t_min && t < closest_t) {
                            closest_t = t;
                            hit = true;
                            best_u = u;  // Store temporarily
                            best_v = v;
                        }
                    }
                }
            }
        } // CRITICAL: Close scope here!
        
        // Fill surface interaction ONCE at the end
        if (hit) {
            si.t = closest_t;
            si.hit = true;
            si.n = normal;
            si.bsdf = bsdf;
            si.Le = Le;
            si.radiosity = radiosity;
            si.p = r.origin() + closest_t * r.direction();
            si.u = best_u;  // Set from stored values
            si.v = best_v;
            return true;
        }
        
        return false;
    }

    Vector3f v00, v10, v11, v01;
    Vector3f bsdf;
    Vector3f normal;
    Vector3f Le;
    Vector3f radiosity;
    Vector3f unshot_rad;
    float area;
    
    float grid[GRID_SIZE];
    Vector3f radiosity_grid[GRID_SIZE];

    Vector3f radiosity_history[RADIOSITY_HISTORY];
    int history_index;
    int history_count;
};

#endif // QUAD_H
