#ifndef PRIMITIVEH
#define PRIMITIVEH

#include "rendering/triangle.h"
#include "rendering/quad.h"
#include <algorithm>
#include <vector>

#ifndef RADIOSITY_HISTORY
#define RADIOSITY_HISTORY 10
#endif

// Primitive type enumeration
enum PrimitiveType {
    PRIM_TRIANGLE = 0,
    PRIM_QUAD = 1
};

// Unified Primitive structure
// Unified Primitive structure
class Primitive {
public:
    PrimitiveType type;
    
    // Union to save memory
    union {
        Triangle tri;
        Quad quad;
    };
    
    // Default constructor
    __host__ __device__ Primitive() : type(PRIM_TRIANGLE) {
        // Initialize triangle by default
    }
    
    // Copy constructor
    __host__ __device__ Primitive(const Primitive& other) : type(other.type) {
        if (type == PRIM_TRIANGLE) {
            tri = other.tri;
        } else {
            quad = other.quad;
        }
    }
    
    // Assignment operator
    __host__ __device__ Primitive& operator=(const Primitive& other) {
        if (this != &other) {
            // Destroy old member
            if (type == PRIM_TRIANGLE) {
                tri.~Triangle();
            } else {
                quad.~Quad();
            }
            
            // Copy new member
            type = other.type;
            if (type == PRIM_TRIANGLE) {
                new (&tri) Triangle(other.tri);
            } else {
                new (&quad) Quad(other.quad);
            }
        }
        return *this;
    }
    
    // Destructor
    __host__ __device__ ~Primitive() {
        // Don't need to explicitly call destructors for POD-like types
        // The union members will be cleaned up automatically
        // Only needed if Triangle/Quad have custom destructors
    }
    
    __host__ __device__ Primitive(const Triangle& t) : type(PRIM_TRIANGLE) {
        new (&tri) Triangle(t);  // Placement new
    }
    
    __host__ __device__ Primitive(const Quad& q) : type(PRIM_QUAD) {
        new (&quad) Quad(q);  // Placement new
    }
    
    // ... rest of the methods remain the same ...

    __host__ __device__ bool intersect(const Ray& r, float t_min, float t_max, 
                                       SurfaceInteractionRecord &si) const {
        if (type == PRIM_TRIANGLE) {
            return tri.intersect(r, t_min, t_max, si);
        } else {
            return quad.intersect(r, t_min, t_max, si);
        }
    }
    
    __host__ __device__ Vector3f centroid() const {
        if (type == PRIM_TRIANGLE) {
            return (tri.v0 + tri.v1 + tri.v2) / 3.0f;
        } else {
            return (quad.v00 + quad.v10 + quad.v11 + quad.v01) * 0.25f;
        }
    }
    
    __host__ __device__ float getArea() const {
        return (type == PRIM_TRIANGLE) ? tri.area : quad.area;
    }
    
    __host__ __device__ Vector3f getNormal() const {
        return (type == PRIM_TRIANGLE) ? tri.normal : quad.normal;
    }
    
    __host__ __device__ Vector3f getBSDF() const {
        return (type == PRIM_TRIANGLE) ? tri.bsdf : quad.bsdf;
    }
    
    __host__ __device__ Vector3f getRadiosity() const {
        return (type == PRIM_TRIANGLE) ? tri.radiosity : quad.radiosity;
    }
    
    __host__ __device__ Vector3f getUnshotRad() const {
        return (type == PRIM_TRIANGLE) ? tri.unshot_rad : quad.unshot_rad;
    }
    
    __host__ __device__ Vector3f getLe() const {
        return (type == PRIM_TRIANGLE) ? tri.Le : quad.Le;
    }
    
    __host__ __device__ void setRadiosity(const Vector3f& rad) {
        if (type == PRIM_TRIANGLE) {
            tri.radiosity = rad;
        } else {
            quad.radiosity = rad;
        }
    }
    
    __host__ __device__ void setUnshotRad(const Vector3f& rad) {
        if (type == PRIM_TRIANGLE) {
            tri.unshot_rad = rad;
        } else {
            quad.unshot_rad = rad;
        }
    }
    
    __host__ __device__ void setLe(const Vector3f& emission) {
        if (type == PRIM_TRIANGLE) {
            tri.Le = emission;
        } else {
            quad.Le = emission;
        }
    }
    
// In bvh.h, replace the sampleUniform method in the Primitive class:

    __host__ __device__ Vector3f sampleUniform(float r1, float r2) const {
        if (type == PRIM_TRIANGLE) {
            // Triangle sampling (barycentric)
            float sqrt_r1 = sqrtf(r1);
            float u = 1.0f - sqrt_r1;
            float v = sqrt_r1 * (1.0f - r2);
            float w = sqrt_r1 * r2;
            return tri.v0 * u + tri.v1 * v + tri.v2 * w;
        } else {
            // Quad sampling - split into two triangles and sample based on area
            // Calculate areas of the two triangles
            Vector3f e1 = quad.v10 - quad.v00;
            Vector3f e2 = quad.v01 - quad.v00;
            float area1 = 0.5f * cross(e1, e2).length();
            
            Vector3f e3 = quad.v11 - quad.v10;
            Vector3f e4 = quad.v11 - quad.v01;
            float area2 = 0.5f * cross(e3, e4).length();
            
            float total_area = area1 + area2;
            float area_ratio = area1 / total_area;
            
            // Choose which triangle to sample based on area
            if (r1 < area_ratio) {
                // Sample triangle 1: v00, v10, v01
                float r1_remapped = r1 / area_ratio;
                float sqrt_r1 = sqrtf(r1_remapped);
                float u = 1.0f - sqrt_r1;
                float v = sqrt_r1 * (1.0f - r2);
                float w = sqrt_r1 * r2;
                return quad.v00 * u + quad.v10 * v + quad.v01 * w;
            } else {
                // Sample triangle 2: v10, v11, v01
                float r1_remapped = (r1 - area_ratio) / (1.0f - area_ratio);
                float sqrt_r1 = sqrtf(r1_remapped);
                float u = 1.0f - sqrt_r1;
                float v = sqrt_r1 * (1.0f - r2);
                float w = sqrt_r1 * r2;
                return quad.v10 * u + quad.v11 * v + quad.v01 * w;
            }
        }
    }

    __host__ __device__ void pushRadiosityHistory(const Vector3f& rad) {
        if (type == PRIM_TRIANGLE) {
            tri.radiosity_history[tri.history_index] = rad;
            tri.history_index = (tri.history_index + 1) % RADIOSITY_HISTORY;
            if (tri.history_count < RADIOSITY_HISTORY) tri.history_count++;
        } else {
            quad.radiosity_history[quad.history_index] = rad;
            quad.history_index = (quad.history_index + 1) % RADIOSITY_HISTORY;
            if (quad.history_count < RADIOSITY_HISTORY) quad.history_count++;
        }
    }

    __host__ __device__ Vector3f getRadiosityHistory(int step) const {
        // step=0 is most recent, step=9 is oldest
        if (type == PRIM_TRIANGLE) {
            if (step >= tri.history_count) return Vector3f(0.0f, 0.0f, 0.0f);
            // Proper circular buffer indexing
            int idx = ((tri.history_index - 1 - step) % RADIOSITY_HISTORY + RADIOSITY_HISTORY) % RADIOSITY_HISTORY;
            return tri.radiosity_history[idx];
        } else {
            if (step >= quad.history_count) return Vector3f(0.0f, 0.0f, 0.0f);
            // Proper circular buffer indexing
            int idx = ((quad.history_index - 1 - step) % RADIOSITY_HISTORY + RADIOSITY_HISTORY) % RADIOSITY_HISTORY;
            return quad.radiosity_history[idx];
        }
    }

    __host__ __device__ Vector3f getRadiosityDelta(int step1, int step2) const {
        return getRadiosityHistory(step1) - getRadiosityHistory(step2);
    }

        
    __host__ __device__ float* getGrid() {
        return (type == PRIM_TRIANGLE) ? tri.grid : quad.grid;
    }
    
    __host__ __device__ Vector3f* getRadiosityGrid() {
        return (type == PRIM_TRIANGLE) ? tri.radiosity_grid : quad.radiosity_grid;
    }
    
    // ================ NEW: Top-K histogram support ================
    
    // Get maximum value in grid
    __host__ float getGridMaxValue() const {
        const float* grid = (type == PRIM_TRIANGLE) ? tri.grid : quad.grid;
        float max_val = 0.0f;
        for (int i = 0; i < GRID_SIZE; i++) {
            max_val = fmaxf(max_val, grid[i]);
        }
        return max_val;
    }
    
    // Find top-K cells by value
    __host__ void getTopKIndices(int k, int* out_indices, float* out_values, int& actual_count) const {
        const float* grid = (type == PRIM_TRIANGLE) ? tri.grid : quad.grid;
        
        if (k <= 0) {
            actual_count = 0;
            return;
        }
        
        // Create temporary pairs of (index, value)
        std::vector<std::pair<int, float>> pairs;
        for (int i = 0; i < GRID_SIZE; i++) {
            pairs.push_back({i, grid[i]});
        }
        
        // Sort in descending order and keep top-K
        std::partial_sort(pairs.begin(), 
                         pairs.begin() + std::min(k, (int)pairs.size()),
                         pairs.end(),
                         [](const auto& a, const auto& b) { return a.second > b.second; });
        
        actual_count = std::min(k, (int)pairs.size());
        for (int i = 0; i < actual_count; i++) {
            out_indices[i] = pairs[i].first;
            out_values[i] = pairs[i].second;
        }
    }
};

#endif
