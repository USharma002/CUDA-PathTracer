#ifndef VECTOR_H
#define VECTOR_H

#include <cmath>
#include <iostream>
#include <numeric>

// Pre-declare the template class
template <typename T, int N>
class Vector;

// Pre-declare template functions to be friends
template <typename T, int N>
inline std::istream& operator>>(std::istream& is, Vector<T, N>& t);

template <typename T, int N>
inline std::ostream& operator<<(std::ostream& os, const Vector<T, N>& t);


template <typename T, int N>
class Vector {
public:
    T e[N];

    // --- Constructors ---
    __host__ __device__ Vector() {
        for (int i = 0; i < N; ++i) e[i] = T(0);
    }
    
    // Constructor with an initializer list
    // Example: Vector<float, 3> my_vec{1.0f, 2.0f, 3.0f};
    __host__ __device__ Vector(const T(&&list)[N]) {
        for (int i = 0; i < N; ++i) e[i] = list[i];
    }
    
    // Allows constructing from a parameter pack
    // Example: Vector<float, 3> my_vec(1.0f, 2.0f, 3.0f);
    template<typename... Args>
    __host__ __device__ Vector(Args... args) : e{static_cast<T>(args)...} {
        static_assert(sizeof...(args) == N, "Incorrect number of initializers for Vector");
    }


    // --- Accessors for 2D, 3D, and 4D vectors ---
    __host__ __device__ inline T x() const { if constexpr (N > 0) return e[0]; }
    __host__ __device__ inline T y() const { if constexpr (N > 1) return e[1]; }
    __host__ __device__ inline T z() const { if constexpr (N > 2) return e[2]; }
    __host__ __device__ inline T w() const { if constexpr (N > 3) return e[3]; }

    __host__ __device__ inline T r() const { if constexpr (N > 0) return e[0]; }
    __host__ __device__ inline T g() const { if constexpr (N > 1) return e[1]; }
    __host__ __device__ inline T b() const { if constexpr (N > 2) return e[2]; }
    __host__ __device__ inline T a() const { if constexpr (N > 3) return e[3]; }

    // --- Operators ---
    __host__ __device__ inline const Vector& operator+() const { return *this; }
    __host__ __device__ inline Vector operator-() const {
        Vector result;
        for (int i = 0; i < N; ++i) result.e[i] = -e[i];
        return result;
    }
    __host__ __device__ inline T operator[](int i) const { return e[i]; }
    __host__ __device__ inline T& operator[](int i) { return e[i]; }

    __host__ __device__ inline Vector& operator+=(const Vector& v2) {
        for (int i = 0; i < N; ++i) e[i] += v2.e[i];
        return *this;
    }

    __host__ __device__ inline Vector& operator-=(const Vector& v2) {
        for (int i = 0; i < N; ++i) e[i] -= v2.e[i];
        return *this;
    }

    __host__ __device__ inline Vector& operator*=(const Vector& v2) {
        for (int i = 0; i < N; ++i) e[i] *= v2.e[i];
        return *this;
    }

    __host__ __device__ inline Vector& operator/=(const Vector& v2) {
        for (int i = 0; i < N; ++i) e[i] /= v2.e[i];
        return *this;
    }

    __host__ __device__ inline Vector& operator*=(const T t) {
        for (int i = 0; i < N; ++i) e[i] *= t;
        return *this;
    }

    __host__ __device__ inline Vector& operator/=(const T t) {
        T k = 1.0 / t;
        for (int i = 0; i < N; ++i) e[i] *= k;
        return *this;
    }

    // --- Vector Math ---
    __host__ __device__ inline T length_squared() const {
        T sum = 0;
        for (int i = 0; i < N; ++i) sum += e[i] * e[i];
        return sum;
    }

    __host__ __device__ inline T length() const {
        return sqrt(length_squared());
    }

    __host__ __device__ inline Vector<T, N> normalized() const {
        T len = length();
        if (len > 0) {
            return *this / len;
        }
        return Vector<T, N>(); // zero vector if length == 0
    }

    __host__ __device__ inline void normalize() {
        T len = length();
        if (len > 0) {
            *this /= len;
        }
    }

    
    // --- Friend functions for stream I/O ---
    friend std::istream& operator>> <T, N>(std::istream& is, Vector<T, N>& t);
    friend std::ostream& operator<< <T, N>(std::ostream& os, const Vector<T, N>& t);
};

// --- Stream Operators Implementation ---
template <typename T, int N>
inline std::istream& operator>>(std::istream& is, Vector<T, N>& t) {
    for (int i = 0; i < N; ++i) is >> t.e[i];
    return is;
}

template <typename T, int N>
inline std::ostream& operator<<(std::ostream& os, const Vector<T, N>& t) {
    for (int i = 0; i < N; ++i) os << t.e[i] << (i == N - 1 ? "" : " ");
    return os;
}

// --- Non-Member Binary Operators ---

// Vector-Vector
template <typename T, int N>
__host__ __device__ inline Vector<T, N> operator+(const Vector<T, N>& v1, const Vector<T, N>& v2) {
    Vector<T, N> result;
    for (int i = 0; i < N; ++i) result.e[i] = v1.e[i] + v2.e[i];
    return result;
}

template <typename T, int N>
__host__ __device__ inline Vector<T, N> operator-(const Vector<T, N>& v1, const Vector<T, N>& v2) {
    Vector<T, N> result;
    for (int i = 0; i < N; ++i) result.e[i] = v1.e[i] - v2.e[i];
    return result;
}

template <typename T, int N>
__host__ __device__ inline Vector<T, N> operator*(const Vector<T, N>& v1, const Vector<T, N>& v2) {
    Vector<T, N> result;
    for (int i = 0; i < N; ++i) result.e[i] = v1.e[i] * v2.e[i];
    return result;
}

template <typename T, int N>
__host__ __device__ inline Vector<T, N> operator/(const Vector<T, N>& v1, const Vector<T, N>& v2) {
    Vector<T, N> result;
    for (int i = 0; i < N; ++i) result.e[i] = v1.e[i] / v2.e[i];
    return result;
}

// Vector-Scalar
template <typename T, int N>
__host__ __device__ inline Vector<T, N> operator*(T t, const Vector<T, N>& v) {
    Vector<T, N> result;
    for (int i = 0; i < N; ++i) result.e[i] = t * v.e[i];
    return result;
}

template <typename T, int N>
__host__ __device__ inline Vector<T, N> operator*(const Vector<T, N>& v, T t) {
    return t * v;
}

template <typename T, int N>
__host__ __device__ inline Vector<T, N> operator/(const Vector<T, N>& v, T t) {
    Vector<T, N> result;
    T k = 1.0 / t;
    for (int i = 0; i < N; ++i) result.e[i] = v.e[i] * k;
    return result;
}

// --- General Vector Functions ---
template <typename T, int N>
__host__ __device__ inline T dot(const Vector<T, N>& v1, const Vector<T, N>& v2) {
    T sum = 0;
    for (int i = 0; i < N; ++i) sum += v1.e[i] * v2.e[i];
    return sum;
}

template <typename T, int N>
__host__ __device__ inline Vector<T, N> unit_vector(Vector<T, N> v) {
    return v / v.length();
}

// --- Specialized 3D Vector Functions ---
template <typename T>
__host__ __device__ inline Vector<T, 3> cross(const Vector<T, 3>& v1, const Vector<T, 3>& v2) {
    return Vector<T, 3>(
        v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1],
        (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
        v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]
    );
}

// --- Common Type Aliases ---
using Vector2f = Vector<float, 2>;
using Vector3f = Vector<float, 3>;
using Vector4f = Vector<float, 4>;

using Vector2d = Vector<double, 2>;
using Vector3d = Vector<double, 3>;
using Vector4d = Vector<double, 4>;

using Vector2i = Vector<int, 2>;
using Vector3i = Vector<int, 3>;
using Vector4i = Vector<int, 4>;

// For color, often represented as a Vector3f
using Spectrum = Vector3f;

#endif // VECTOR_H