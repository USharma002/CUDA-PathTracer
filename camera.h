#ifndef CAMERAH
#define CAMERAH

#include "vec3.h"
#include "ray.h"

class camera
{
public:
    __host__ __device__  camera() {}

    __host__ __device__  camera(
        const vec3& position,
        int image_width,
        int image_height,
        float focal_length,
        float viewport_height,
        float viewport_width
    )
        : position(position),
          focal_length(focal_length),
          viewport_height(viewport_height),
          viewport_width(viewport_width),
          image_width(image_width),
          image_height(image_height)
    {
        // Define viewport basis vectors (u is horizontal, v is vertical)
        viewport_u = vec3(viewport_width, 0, 0);
        viewport_v = vec3(0, -viewport_height, 0);  // y downwards to match image coords

        pixel_delta_u = viewport_u / float(image_width);
        pixel_delta_v = viewport_v / float(image_height);

        // Calculate upper left corner of viewport in world space
        viewport_upper_left = position - vec3(0, 0, focal_length) - viewport_u * 0.5f - viewport_v * 0.5f;

        // Center of the top-left pixel
        pixel00_loc = viewport_upper_left + 0.5f * (pixel_delta_u + pixel_delta_v);
    }

    // Generate a ray for pixel (u, v)
    __host__ __device__  ray get_ray(int u, int v) const
    {
        // Calculate pixel center location in world space
        vec3 pixel_center = pixel00_loc + float(u) * pixel_delta_u + float(v) * pixel_delta_v;
        // Ray direction from camera position to pixel center
        vec3 ray_direction = pixel_center - position;
        return ray(position, unit_vector(ray_direction));
    }

public:
    vec3 position;

    float focal_length;
    float viewport_height;
    float viewport_width;

    int image_width;
    int image_height;

    vec3 viewport_u;
    vec3 viewport_v;

    vec3 pixel_delta_u;
    vec3 pixel_delta_v;

    vec3 viewport_upper_left;
    vec3 pixel00_loc;
};

#endif
