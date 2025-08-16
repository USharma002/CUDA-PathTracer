#ifndef CAMERAH
#define CAMERAH

#include "ray.h"
#include "vec3.h"

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

class Sensor {
    public:
        __host__ __device__ Sensor(Vector3f lookfrom, Vector3f lookat, Vector3f vup, float vfov, float aspect) {
            // vfov is top to bottom in degrees
            Vector3f u, v, w;
            float theta = vfov*M_PI/180;
            float half_height = tan(theta/2);
            float half_width = aspect * half_height;
            origin = lookfrom;
            w = unit_vector(lookfrom - lookat);
            u = unit_vector(cross(vup, w));
            v = cross(w, u);
            lower_left_corner = origin - half_width*u -half_height*v - w;
            horizontal = 2*half_width*u;
            vertical = 2*half_height*v;
        }
        __device__ Ray get_ray(float u, float v) { return Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin); }

        Vector3f origin;
        Vector3f lower_left_corner;
        Vector3f horizontal;
        Vector3f vertical;
        float image_width;
        float image_height;
};

#endif