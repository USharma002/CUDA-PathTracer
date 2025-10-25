#ifndef SURFACE_INTERACTION_RECORD_H
#define SURFACE_INTERACTION_RECORD_H

#include "vector.h"

// Forward declarations
class Triangle;
class Quad;
class Primitive;

enum HitType {
    HIT_NONE = 0,
    HIT_TRIANGLE = 1,
    HIT_QUAD = 2
};

struct SurfaceInteractionRecord {
    bool hit;
    Vector3f p;        // Hit point
    Vector3f n;        // Surface normal at hit
    Vector3f bsdf;     // BSDF/reflectance
    Vector3f Le;       // Emitted radiance
    Vector3f radiosity; // Precomputed radiosity
    float t;           // Ray parameter
    float u, v;        // Surface parameters (barycentric for triangles, bilinear for quads)
    
    // Primitive information
    HitType hit_type;
    const Primitive* prim_ptr;  // Unified primitive pointer
    
    // Backwards compatibility pointers (deprecated, use prim_ptr instead)
    const Triangle* triangle_ptr;
    
    __host__ __device__ SurfaceInteractionRecord() 
        : hit(false), 
          p(Vector3f(0, 0, 0)),
          n(Vector3f(0, 0, 1)),
          bsdf(Vector3f(0, 0, 0)),
          Le(Vector3f(0, 0, 0)),
          radiosity(Vector3f(0, 0, 0)),
          t(0), 
          u(0), 
          v(0),
          hit_type(HIT_NONE),
          prim_ptr(nullptr),
          triangle_ptr(nullptr) {}
    
    // Helper methods to check primitive type
    __host__ __device__ bool isTriangle() const { return hit_type == HIT_TRIANGLE; }
    __host__ __device__ bool isQuad() const { return hit_type == HIT_QUAD; }
};

#endif