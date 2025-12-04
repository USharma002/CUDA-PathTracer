#ifndef CAMERAH
#define CAMERAH

#include "core/ray.h"
#include "core/vector.h"

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

#define ToRadian(x) (float)(((x) * M_PI / 180.0f))
#define ToDegree(x) (float)(((x) * 180.0f / M_PI))

class Sensor {
public:
    __host__ __device__ Sensor(Vector3f lookfrom, Vector3f lookat, Vector3f vup, float vfov, float aspect) {
        // vfov is top to bottom in degrees
        origin = lookfrom;
        this->vup = vup;
        this->lookat = lookat;
        this->vfov = vfov;
        this->aspect = aspect;
        this->radius = (lookfrom - lookat).length();
        this->yaw = 90.0f;
        this->pitch = 0.0f;

        // Initial calculation of camera state
        updateCamera();
    }

    __device__ Ray get_ray(float u, float v) {
        return Ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);
    }

    // *** MODIFIED ***
    // This function now ONLY updates the view plane based on the current state.
    // It no longer recalculates the origin.
    __host__ __device__ void updateCamera() {
        // vfov is top to bottom in degrees
        Vector3f u, v, w;
        float theta = vfov * M_PI / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width = aspect * half_height;

        w = unit_vector(origin - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width * u - half_height * v - w;
        horizontal = 2 * half_width * u;
        vertical = 2 * half_height * v;
    }

    // *** NEW/REPURPOSED FUNCTION ***
    // This function is now responsible for updating the camera's orbital position.
    // It should be called from the main loop AFTER yaw/pitch have been changed.
    __host__ void updateCameraOrbit() {
        float yawRad = ToRadian(yaw);
        float pitchRad = ToRadian(pitch);

        // Keep the position on a sphere of a given radius around the lookat point
        origin.e[0] = lookat.x() + radius * cosf(pitchRad) * cosf(yawRad);
        origin.e[1] = lookat.y() + radius * sinf(pitchRad);
        origin.e[2] = lookat.z() + radius * cosf(pitchRad) * sinf(yawRad);

        // After updating the origin, we must recalculate the view plane
        updateCamera();
    }

    __host__ void setPosition(const Vector3f& new_pos) {
        origin = new_pos;
        updateCamera();
    }

    __host__ void setLookAt(const Vector3f& new_target) {
        lookat = new_target;
        updateCamera();
    }

    __host__ void setFov(float new_vfov) {
        vfov = new_vfov;
        updateCamera();
    }

    float radius;
    float yaw;
    float pitch;

    Vector3f vup;
    Vector3f lookat;

    Vector3f origin;
    Vector3f lower_left_corner;
    Vector3f horizontal;
    Vector3f vertical;

    float vfov;
    float aspect;

    int image_width;
    int image_height;
};

#endif