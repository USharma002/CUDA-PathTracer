# Rendering Engine Documentation (CUDA/C++)

---

## Core Math / Utilities

### class `Vector3f`

Represents a 3D vector with basic arithmetic and utility functions.

**Constructor:**

```cpp
Vector3f(float x, float y, float z);
```

**Attributes:**

* `float x` — X component
* `float y` — Y component
* `float z` — Z component

**Methods:**

```cpp
Vector3f operator+(const Vector3f& other) const;   // Vector addition
Vector3f operator-(const Vector3f& other) const;   // Vector subtraction
Vector3f operator*(const Vector3f& other) const;   // Component-wise multiplication
Vector3f operator/(const Vector3f& other) const;   // Component-wise division
float dot(const Vector3f& other) const;            // Dot product
Vector3f cross(const Vector3f& other) const;       // Cross product
float length() const;                              // Magnitude of the vector
float length_squared() const;                      // Magnitude square of the vector
Vector3f normalize() const;                        // Returns normalized vector
```

<b>TODO</b>
```
Vector3f clamp(float min, float max) const;       // Clamps each component
Vector3f reflect(const Vector3f& normal) const;   // Reflection vector
Vector3f refract(const Vector3f& normal, float ior) const;  // Refraction vector
```

---

## Rays & Geometry

### class `Ray`

Represents a ray in 3D space.

**Constructor:**

```cpp
Ray(const Vector3f& origin, const Vector3f& dir, float tmin = 0.0f, float tmax = FLT_MAX);
```

**Attributes:**

* `Vector3f o` — Ray origin
* `Vector3f d` — Normalized direction
**TODO**
* `float tmin` — Minimum valid distance
* `float tmax` — Maximum valid distance

**Methods:**

```cpp
Vector3f at(float t) const;    // Returns o + t * d
```

---

### class `SurfaceInteraction`

Stores information about a ray-surface intersection.

**Attributes:**

* `bool hit` — Valid hit or not
* `Vector3f p` — Hit point
* `Vector3f n` — Shading normal
* `float t` — Distance along the ray

**TODO**
* `Material* material` — Material at intersection
* `Vector2f uv` — Texture coordinates
* `Vector3f wo` — Outgoing direction (-ray.dir)

**Methods:**

```cpp
bool is_valid() const;   // Returns true if intersection is valid
```

---

### class `Triangle`

Represents a triangle in 3D space.

**Attributes:**

* `Vector3f v0, v1, v2` — Vertices
* `Material* material` — Material
**TODO**
* `Vector2f uv0, uv1, uv2` — Texture coordinates
* `Vector3f n0, n1, n2` — Vertex normals (optional)

**Methods:**

```cpp
bool intersect(const Ray& ray, SurfaceInteraction& si) const;  // Ray-triangle intersection
```
**TODO**
```
AABB get_bounds() const;                                        // Returns bounding box
Vector3f sample_point(const Sample2D& sample) const;            // Sample a point on triangle
```

---

## Materials & BSDF

### class `Material`

Abstract BSDF class.

**Attributes:**

* `enum Type` — Diffuse, Mirror, Glass, etc.
* `float albedo, roughness, ior` — Material parameters

**Methods:**

```cpp
Vector3f f(const SurfaceInteraction& si, const Vector3f& wi, const Vector3f& wo) const;   // BSDF evaluation
Vector3f sample(const SurfaceInteraction& si, const Vector3f& wo, RNG& rng) const;         // Sampled direction and BSDF value
float pdf(const SurfaceInteraction& si, const Vector3f& wi, const Vector3f& wo) const;      // PDF for given direction
Vector3f Le(const SurfaceInteraction& si, const Vector3f& wo) const;                        // Emitted radiance
```

---

## Scene & Camera

### class `Sensor`

Camera class that generates rays for each pixel.

**Attributes:**

* `Vector3f pos, look_at, up` — Camera position and orientation
* `float fov` — Field of view
* `float aspect_ratio` — Image aspect ratio

**TODO**
* `Film film` — Pixel buffer

**Methods:**

```cpp
Ray generate_ray(float x, float y, RNG& rng) const;   // Generate camera ray for pixel
```

---

### class `Scene`

Contains geometry, emitters, and acceleration structures.

**Attributes:**

* `std::vector<Triangle> shapes` — Scene geometry

**TODO**
* `std::vector<Emitter*> emitters` — Light sources
* `AccelStructure* accel` — Acceleration structure (BVH/kd-tree)
* `Environment* env_light` — Optional environment light

**Methods:**

```cpp
bool intersect(const Ray& ray, SurfaceInteraction& si) const;  // Scene intersection
Emitter* sample_emitter(RNG& rng) const;                        // Random emitter selection
bool visible(const Vector3f& p1, const Vector3f& p2) const;    // Shadow ray check
```

---

## Rendering

### class `Integrator`

Path tracing integrator.

**Attributes:**

* `int max_depth` — Maximum recursion depth
* `float rr_threshold` — Russian roulette probability

**Methods:**

```cpp
Vector3f Li(const Ray& ray, const Scene& scene, RNG& rng, int depth = 0) const;   // Compute radiance recursively
void render(const Scene& scene, Sensor& sensor);                                   // Loop over pixels and compute image
```

---

## Supporting Infrastructure

### class `FileManager`

Loads geometry and scene descriptions.

**Methods:**

```cpp
std::vector<Triangle> load_obj(const std::string& filepath) const;   // Load OBJ file
Scene load_scene(const std::string& filepath) const;                  // Parse scene description
```

### `stb_image_write.h`

Header-only library for writing images.

### `math_utils.h`

Utility math functions.

**Methods:**

**TODO**
```cpp
Vector3f sample_hemisphere();
Vector3f sample_cosine_hemisphere();
Vector3f sample_sphere();
Vector3f world_to_local(const Vector3f& v, const Vector3f& normal);
Vector3f local_to_world(const Vector3f& v, const Vector3f& normal);
```
