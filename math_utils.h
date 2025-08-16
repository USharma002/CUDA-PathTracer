#define RANDVector3f Vector3f(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

#ifndef FLT_MAX
#    define FLT_MAX 2e+30
#endif

#ifndef M_PI
#define M_PI        3.14159265358979323846f
#endif

#define M_2PI       (2.0f * M_PI)
#define M_4PI       (4.0f * M_PI)

#define M_INV_PI    (1.0f / M_PI)
#define M_INV_2PI   (1.0f / (2.0f * M_PI))
#define M_INV_4PI   (1.0f / (4.0f * M_PI))

__device__ Vector3f random_in_unit_sphere(curandState *local_rand_state) {
    Vector3f p;
    do {
        p = 2.0f*RANDVector3f - Vector3f(1,1,1);
    } while (p.length_squared() >= 1.0f);
    return p;
}

__host__ __device__ inline float safe_sqrt(float x) {
    return sqrtf(fmaxf(0.0f, x));
}

__host__ __device__ inline double safe_sqrt(double x) {
    return sqrtf(fmax(0.0f, x));
}

__host__ __device__ inline float safe_acos(float x) {
    return acosf(fmaxf(-1.0f, fminf(1.0f, x)));
}

__host__ __device__ inline double safe_acos(double x) {
    return acos(fmax(-1.0, fmin(1.0, x)));
}

template <typename T>
__host__ __device__ inline T clamp(T v, T lo, T hi) {
    return (v < lo) ? lo : (v > hi ? hi : v);
}

__host__ __device__ Vector2f cartesianToSpherical(const Vector3f &v) {
	/* caution! acos(val) produces NaN when val is out of [-1, 1]. */
	const Vector3f vn = v.normalized();
	Vector2f sph{ safe_acos(vn[2]), atan2(vn[1], vn[0]) };
	if (sph[1] < 0)
		sph[1] += M_2PI;
	return sph;
}

__host__ __device__ Vector2f cartesianToSphericalNormalized(const Vector3f &v) {
	Vector2f sph = cartesianToSpherical(v);
	return { float(sph[0] / M_PI), float(sph[1] / M_2PI) };
}

__host__ __device__ Vector3f sphericalToCartesian(float theta, float phi) {
	float sinTheta = sin(theta);
	float cosTheta = cos(theta);
	float sinPhi   = sin(phi);
	float cosPhi   = cos(phi);
	return {
		sinTheta * cosPhi,
		sinTheta * sinPhi,
		cosTheta,
	};
}

__host__ __device__  Vector3f sphericalToCartesian(float sinTheta, float cosTheta, float phi) {
	return Vector3f(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

__host__ __device__ Vector3f sphericalToCartesian(const Vector2f sph) {
	return sphericalToCartesian(sph[0], sph[1]);
}

__host__ __device__ float sphericalTheta(const Vector3f &v) {
    return acos(clamp(v[2], -1.f, 1.f));
}

__host__ __device__ float sphericalPhi(const Vector3f &v) {
	float p = atan2(v[1], v[0]);
	return (p < 0) ? (p + 2 * M_PI) : p;
}

__host__ __device__ Vector3f squareToUniformSphere(Vector2f u) {
	float z = 1 - 2 * u[0];
	float r = safe_sqrt(1 - z * z);
	float phi = M_2PI * u[1];
	return Vector3f(r * cos(phi), r * sin(phi), z);
}

__host__ __device__ Vector2f uniformSphereToSquare(Vector3f v) {
	const Vector3f vn = v.normalized();
	float phi = atan2(vn[1], vn[0]) * M_INV_2PI;
	return Vector2f(
		(1.f - vn[2]) * 0.5f,
		(phi < 0.f) ? phi + 1.f : phi
	);
}

__host__ __device__ float squareToUniformSpherePdf() { 
    return M_INV_4PI; 
}

__host__ __device__ float squareToUniformSpherePdfInv() {
    return M_4PI; 
}

__host__ __device__ Vector3f squareToUniformHemisphere(Vector2f u) {
	float z = u[0];
	float r = safe_sqrt(1 - z * z);
	float phi = M_2PI * u[1];
	return Vector3f(r * cos(phi), r * sin(phi), z);
}

__host__ __device__ Vector2f uniformHemisphereToSquare(Vector3f v) {
	const Vector3f vn = v.normalized();
	float phi = atan2(vn[1], vn[0]) * M_INV_2PI;
	return Vector2f(
		vn[2],
		(phi < 0.f) ? phi + 1.f : phi
	);
}

__host__ __device__ float squareToUniformHemispherePdf() { 
    return M_INV_2PI; 
}

__host__ __device__ float squareToUniformHemispherePdfInv() { 
    return M_2PI; 
}