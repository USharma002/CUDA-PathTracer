# Rendering Engine Overview

## 🔹 Core Math / Utilities
**vec3.h**

**Attributes:**  
- `x, y, z` (float/double)

**Methods:**  
- Basic operations: `+`, `-`, `*`, `/`  
- `dot`, `cross`  
- `length`, `normalize`  
- Utility: `clamp`, `reflect`, `refract`  

---

## 🔹 Rays & Geometry
**ray.h**

**Attributes:**  
- `origin` (`vec3`)  
- `dir` (`vec3`, normalized)  
- `tmin`, `tmax` (float)  

**Methods:**  
- `point_at(t)` → returns `origin + t * dir`  

**surface_interaction_record.h**

**Attributes:**  
- `p` (hit point)  
- `n` (shading normal)  
- `uv` (texture coordinates)  
- `t` (distance along ray)  
- `wo` (outgoing direction = `-ray.dir`)  
- `material / bsdf` (pointer/ref to material at hit)  

**Methods:**  
- `bool is_valid()`  

**triangle.h**

**Attributes:**  
- `v0, v1, v2` (vertices)  
- `n0, n1, n2` (normals, optional)  
- `uv0, uv1, uv2`  
- `material` (pointer to material)  
- `Le` (emission color, optional)  

**Methods:**  
- `bool intersect(ray, SurfaceInteraction&)`  
- `get_bounds()` (for BVH)  
- `sample_point(Sample2D)` (for area light sampling)  

---

## 🔹 Materials & BSDF
**material.h** (BSDF abstraction)

**Attributes:**  
- `Type` (diffuse, mirror, glass…)  
- Parameters: `albedo`, `roughness`, `ior`, etc.  

**Methods:**  
- `f(si, wi, wo)` → BSDF value  
- `sample(si, wo, rng)` → returns sampled `wi`, `pdf`, `f`  
- `pdf(si, wi, wo)`  
- `Le(si, wo)` → emitted radiance if emissive  

---

## 🔹 Scene & Camera
**sensor.h** (Camera)

**Attributes:**  
- `pos`, `look_at`, `up`  
- `fov` (field of view)  
- `aspect_ratio`  
- `film` (pixel buffer)  

**Methods:**  
- `Ray generate_ray(x, y, rng)` → for pixel sample  

**scene.h**

**Attributes:**  
- `std::vector<Triangle> shapes`  
- `std::vector<Emitter*> emitters`  
- Acceleration structure (BVH, kd-tree)  
- Optional environment light  

**Methods:**  
- `bool intersect(ray, SurfaceInteraction&)`  
- `Emitter* sample_emitter(rng)`  
- `bool visible(p1, p2)` (shadow ray check)  

---

## 🔹 Rendering
**integrator.h**

**Attributes:**  
- `max_depth`  
- `rr_threshold` (Russian roulette probability)  

**Methods:**  
- `Li(ray, scene, rng)` → recursively compute radiance  
- `render(scene, sensor)` → loops over pixels, calls `Li`  

---

## 🔹 Supporting Infrastructure
**file_manager.h**

**Attributes:** none  

**Methods:**  
- `load_obj(filepath)` → returns list of triangles + materials  
- `load_scene(filepath)` → parse scene description  

**stb_image_write.h**  
- 3rd party header-only library, already included  

**math_utils.h**  

**Methods / Utilities:**  
- Random sampling functions: `sample_hemisphere`, `sample_cosine_hemisphere`, `sample_sphere`  
- Coordinate transforms: world ↔ local shading frame  
