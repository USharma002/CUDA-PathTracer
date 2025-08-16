#define RANDVector3f Vector3f(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

#ifndef FLT_MAX
#    define FLT_MAX 2e+30
#endif

__device__ Vector3f random_in_unit_sphere(curandState *local_rand_state) {
    Vector3f p;
    do {
        p = 2.0f*RANDVector3f - Vector3f(1,1,1);
    } while (p.length_squared() >= 1.0f);
    return p;
}
