// onb.h
#ifndef ONBH
#define ONBH

#include "vector.h"

// Orthonormal Basis (local coordinate system)
struct Onb {
    __device__ void build_from_w(const Vector3f& n) {
        axis[2] = unit_vector(n); // W axis is the normal
        Vector3f a = (fabs(w().x()) > 0.9f) ? Vector3f(0, 1, 0) : Vector3f(1, 0, 0);
        axis[0] = unit_vector(cross(a, w())); // U axis
        axis[1] = cross(w(), u());           // V axis
    }

    __device__ Vector3f local(const Vector3f& a) const {
        return a.x() * u() + a.y() * v() + a.z() * w();
    }

    __device__ Vector3f u() const { return axis[0]; }
    __device__ Vector3f v() const { return axis[1]; }
    __device__ Vector3f w() const { return axis[2]; }

    Vector3f axis[3];
};

#endif