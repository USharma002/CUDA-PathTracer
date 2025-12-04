/**
 * @file pbrt_loader.h
 * @brief PBRT scene file loader
 * 
 * Loads PBRT v3/v4 scene files and converts them to our primitive format.
 * Supports various PBRT material types and converts them to our simple BSDF model.
 * 
 * Note: This is HOST-ONLY code - not for CUDA device compilation.
 */

#ifndef PBRT_LOADER_H
#define PBRT_LOADER_H

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <filesystem>

#include <pbrtParser/Scene.h>

#include "core/vector.h"
#include "rendering/primitive.h"
#include "rendering/triangle.h"
#include "rendering/quad.h"

namespace fs = std::filesystem;

// ============================================================================
// PBRT MATERIAL CONVERSION
// ============================================================================

struct PBRTMaterial {
    Vector3f diffuse = Vector3f(0.8f, 0.8f, 0.8f);
    Vector3f specular = Vector3f(0.0f, 0.0f, 0.0f);
    Vector3f emission = Vector3f(0.0f, 0.0f, 0.0f);
    float roughness = 0.5f;
    float metallic = 0.0f;
    float ior = 1.5f;
    float transmission = 0.0f;
    
    // Return simple BSDF color (diffuse + some specular influence)
    Vector3f getBSDF() const {
        // Blend diffuse and specular based on metallic
        return diffuse * (1.0f - metallic) + specular * metallic;
    }
};

// Helper: Convert pbrt vec3f to our Vector3f
inline Vector3f toVector3f(const pbrt::vec3f& v) {
    return Vector3f(v.x, v.y, v.z);
}

// Helper: Convert pbrt vec3i to int array
inline void toIndices(const pbrt::vec3i& v, int* out) {
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
}

// Helper: Apply transform to a point
inline Vector3f transformPoint(const pbrt::affine3f& xfm, const pbrt::vec3f& p) {
    pbrt::vec3f result;
    result.x = xfm.l.vx.x * p.x + xfm.l.vy.x * p.y + xfm.l.vz.x * p.z + xfm.p.x;
    result.y = xfm.l.vx.y * p.x + xfm.l.vy.y * p.y + xfm.l.vz.y * p.z + xfm.p.y;
    result.z = xfm.l.vx.z * p.x + xfm.l.vy.z * p.y + xfm.l.vz.z * p.z + xfm.p.z;
    return toVector3f(result);
}

// Helper: Apply transform to a normal (inverse transpose of upper 3x3)
inline Vector3f transformNormal(const pbrt::affine3f& xfm, const pbrt::vec3f& n) {
    // For normals, we need inverse transpose, but for orthonormal transforms
    // the upper 3x3 works. For simplicity, just transform and renormalize.
    pbrt::vec3f result;
    result.x = xfm.l.vx.x * n.x + xfm.l.vy.x * n.y + xfm.l.vz.x * n.z;
    result.y = xfm.l.vx.y * n.x + xfm.l.vy.y * n.y + xfm.l.vz.y * n.z;
    result.z = xfm.l.vx.z * n.x + xfm.l.vy.z * n.y + xfm.l.vz.z * n.z;
    return unit_vector(toVector3f(result));
}

// ============================================================================
// MATERIAL CONVERSION FUNCTIONS
// ============================================================================

inline PBRTMaterial convertMaterial(pbrt::Material::SP mat) {
    PBRTMaterial result;
    
    if (!mat) return result;
    
    // Disney Material
    if (auto m = std::dynamic_pointer_cast<pbrt::DisneyMaterial>(mat)) {
        result.diffuse = toVector3f(m->color);
        result.roughness = m->roughness;
        result.metallic = m->metallic;
        result.ior = m->eta;
        result.specular = result.diffuse * m->metallic;
    }
    // Matte Material (pure diffuse)
    else if (auto m = std::dynamic_pointer_cast<pbrt::MatteMaterial>(mat)) {
        result.diffuse = toVector3f(m->kd);
        result.roughness = 1.0f;
        result.metallic = 0.0f;
    }
    // Plastic Material
    else if (auto m = std::dynamic_pointer_cast<pbrt::PlasticMaterial>(mat)) {
        result.diffuse = toVector3f(m->kd);
        result.specular = toVector3f(m->ks);
        result.roughness = sqrtf(m->roughness); // PBRT uses squared roughness
    }
    // Metal Material
    else if (auto m = std::dynamic_pointer_cast<pbrt::MetalMaterial>(mat)) {
        // Convert eta/k to reflectivity at normal incidence
        Vector3f eta = toVector3f(m->eta);
        Vector3f k = toVector3f(m->k);
        // Fresnel reflectance at normal incidence: ((n-1)^2 + k^2) / ((n+1)^2 + k^2)
        Vector3f r;
        for (int i = 0; i < 3; i++) {
            float n = eta[i], kv = k[i];
            r[i] = ((n-1)*(n-1) + kv*kv) / ((n+1)*(n+1) + kv*kv);
        }
        result.diffuse = r;
        result.metallic = 1.0f;
        result.roughness = sqrtf((m->uRoughness + m->vRoughness) * 0.5f);
    }
    // Mirror Material
    else if (auto m = std::dynamic_pointer_cast<pbrt::MirrorMaterial>(mat)) {
        result.diffuse = Vector3f(0.0f, 0.0f, 0.0f);
        result.specular = toVector3f(m->kr);
        result.metallic = 1.0f;
        result.roughness = 0.0f;
    }
    // Glass Material
    else if (auto m = std::dynamic_pointer_cast<pbrt::GlassMaterial>(mat)) {
        result.diffuse = toVector3f(m->kt);
        result.transmission = 1.0f;
        result.ior = m->index;
        result.roughness = 0.0f;
    }
    // Substrate Material
    else if (auto m = std::dynamic_pointer_cast<pbrt::SubstrateMaterial>(mat)) {
        result.diffuse = toVector3f(m->kd);
        result.specular = toVector3f(m->ks);
        result.roughness = sqrtf((m->uRoughness + m->vRoughness) * 0.5f);
    }
    // Uber Material
    else if (auto m = std::dynamic_pointer_cast<pbrt::UberMaterial>(mat)) {
        result.diffuse = toVector3f(m->kd);
        result.specular = toVector3f(m->ks);
        result.roughness = sqrtf(m->roughness);
        result.transmission = (m->kt.x + m->kt.y + m->kt.z) / 3.0f;
    }
    // Translucent Material
    else if (auto m = std::dynamic_pointer_cast<pbrt::TranslucentMaterial>(mat)) {
        result.diffuse = toVector3f(m->kd);
        result.transmission = 0.5f;
    }
    // Fallback for unknown materials
    else {
        std::cout << "  [PBRT] Unknown material type: " << mat->toString() << std::endl;
    }
    
    return result;
}

// ============================================================================
// MAIN PBRT LOADER
// ============================================================================

/**
 * Load a PBRT scene file and convert to primitives
 * 
 * @param pbrt_filename Path to the .pbrt file
 * @param h_list_out Output array of primitives (allocated by this function)
 * @param num_primitives_out Output number of primitives
 * @return true on success, false on failure
 */
static bool loadPBRT(const std::string& pbrt_filename, Primitive** h_list_out, int& num_primitives_out) {
    std::cout << "\n========== LOADING PBRT SCENE ==========" << std::endl;
    std::cout << "File: " << pbrt_filename << std::endl;
    
    // Parse the PBRT file
    pbrt::Scene::SP scene;
    try {
        scene = pbrt::importPBRT(pbrt_filename);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing PBRT file: " << e.what() << std::endl;
        return false;
    }
    
    if (!scene) {
        std::cerr << "Error: Failed to load PBRT scene from " << pbrt_filename << std::endl;
        return false;
    }
    
    // Flatten scene hierarchy
    scene->makeSingleLevel();
    
    std::string basepath = fs::path(pbrt_filename).parent_path().string();

    // temporary storage for primitives (used either for full load or proxy)
    std::vector<Primitive> temp_primitives;

    // Safety: estimate total triangles before expanding geometry to avoid OOM/overload
    const size_t PBRT_MAX_TRIANGLES = 2000000; // configurable threshold
    size_t estimated_tris = 0;
    // sum instances
    for (const auto &inst : scene->world->instances) {
        for (const auto &shape : inst->object->shapes) {
            if (auto mesh = std::dynamic_pointer_cast<pbrt::TriangleMesh>(shape)) {
                estimated_tris += mesh->index.size();
                if (estimated_tris > PBRT_MAX_TRIANGLES) break;
            }
        }
        if (estimated_tris > PBRT_MAX_TRIANGLES) break;
    }
    // sum world shapes
    if (estimated_tris <= PBRT_MAX_TRIANGLES) {
        for (const auto &shape : scene->world->shapes) {
            if (auto mesh = std::dynamic_pointer_cast<pbrt::TriangleMesh>(shape)) {
                estimated_tris += mesh->index.size();
                if (estimated_tris > PBRT_MAX_TRIANGLES) break;
            }
        }
    }

    if (estimated_tris > PBRT_MAX_TRIANGLES) {
        std::cerr << "Warning: PBRT scene too large (" << estimated_tris << " triangles) - creating bounding-box proxy instead of expanding geometry." << std::endl;
        // Create a simple axis-aligned bounding box proxy so something is visible
        pbrt::box3f pb = scene->getBounds();
        pbrt::vec3f bmin = pb.lower;
        pbrt::vec3f bmax = pb.upper;
        Vector3f v000(bmin.x, bmin.y, bmin.z);
        Vector3f v001(bmin.x, bmin.y, bmax.z);
        Vector3f v010(bmin.x, bmax.y, bmin.z);
        Vector3f v011(bmin.x, bmax.y, bmax.z);
        Vector3f v100(bmax.x, bmin.y, bmin.z);
        Vector3f v101(bmax.x, bmin.y, bmax.z);
        Vector3f v110(bmax.x, bmax.y, bmin.z);
        Vector3f v111(bmax.x, bmax.y, bmax.z);

        std::vector<Triangle> boxTris;
        auto addQuad = [&](const Vector3f &a, const Vector3f &b, const Vector3f &c, const Vector3f &d){
            Vector3f n = unit_vector(cross(b - a, c - a));
            boxTris.emplace_back(a, b, c, Vector3f(0.8f,0.2f,0.2f), n);
            boxTris.emplace_back(a, c, d, Vector3f(0.8f,0.2f,0.2f), n);
        };
        // -X face
        addQuad(v000, v001, v011, v010);
        // +X face
        addQuad(v100, v110, v111, v101);
        // -Y face
        addQuad(v000, v100, v101, v001);
        // +Y face
        addQuad(v010, v011, v111, v110);
        // -Z face
        addQuad(v000, v010, v110, v100);
        // +Z face
        addQuad(v001, v101, v111, v011);

        temp_primitives.reserve(boxTris.size());
        for (auto &t : boxTris) temp_primitives.push_back(Primitive(t));

        num_primitives_out = (int)temp_primitives.size();
        *h_list_out = new Primitive[num_primitives_out];
        for (int i = 0; i < num_primitives_out; ++i) (*h_list_out)[i] = temp_primitives[i];

        std::cout << "========== PBRT SCENE LOADED (proxy) ==========" << std::endl;
        std::cout << "  Proxy Triangles: " << num_primitives_out << std::endl;
        std::cout << "========================================" << std::endl;
        return true;
    }
    
    // Cache for converted materials
    std::map<pbrt::Material::SP, PBRTMaterial> material_cache;
    
    // Collect all primitives
    int num_triangles = 0;
    int num_meshes = 0;
    
    // Process all instances
    for (const auto& inst : scene->world->instances) {
        pbrt::affine3f transform = inst->xfm;
        
        for (const auto& shape : inst->object->shapes) {
            // Only handle triangle meshes
            auto mesh = std::dynamic_pointer_cast<pbrt::TriangleMesh>(shape);
            if (!mesh) {
                std::cout << "  [PBRT] Skipping non-triangle shape: " << shape->toString() << std::endl;
                continue;
            }
            
            num_meshes++;
            
            // Get material
            PBRTMaterial mat;
            if (mesh->material) {
                if (material_cache.count(mesh->material)) {
                    mat = material_cache[mesh->material];
                } else {
                    mat = convertMaterial(mesh->material);
                    material_cache[mesh->material] = mat;
                }
            }
            
            // Get emission from area light
            Vector3f emission(0.0f, 0.0f, 0.0f);
            if (mesh->areaLight) {
                if (auto light = std::dynamic_pointer_cast<pbrt::DiffuseAreaLightRGB>(mesh->areaLight)) {
                    emission = toVector3f(light->L);
                } else if (auto light = std::dynamic_pointer_cast<pbrt::DiffuseAreaLightBB>(mesh->areaLight)) {
                    emission = toVector3f(light->LinRGB());
                }
            }
            
            // Process triangles
            int num_faces = mesh->index.size();
            bool has_normals = mesh->normal.size() >= mesh->vertex.size();
            
            for (int i = 0; i < num_faces; i++) {
                const pbrt::vec3i& idx = mesh->index[i];
                
                // Get vertices and transform
                Vector3f v0 = transformPoint(transform, mesh->vertex[idx.x]);
                Vector3f v1 = transformPoint(transform, mesh->vertex[idx.y]);
                Vector3f v2 = transformPoint(transform, mesh->vertex[idx.z]);
                
                // Get or compute normal
                Vector3f normal;
                if (has_normals) {
                    normal = transformNormal(transform, mesh->normal[idx.x]);
                } else {
                    normal = unit_vector(cross(v1 - v0, v2 - v0));
                }
                
                // Create triangle
                Triangle tri(v0, v1, v2, mat.getBSDF(), normal);
                tri.Le = emission;
                
                temp_primitives.push_back(Primitive(tri));
                num_triangles++;
            }
        }
    }
    
    // Also check for shapes directly in the world (not instanced)
    for (const auto& shape : scene->world->shapes) {
        auto mesh = std::dynamic_pointer_cast<pbrt::TriangleMesh>(shape);
        if (!mesh) continue;
        
        num_meshes++;
        
        // Identity transform for world shapes
        pbrt::affine3f identity;
        identity.l.vx = pbrt::vec3f(1, 0, 0);
        identity.l.vy = pbrt::vec3f(0, 1, 0);
        identity.l.vz = pbrt::vec3f(0, 0, 1);
        identity.p = pbrt::vec3f(0, 0, 0);
        
        PBRTMaterial mat;
        if (mesh->material) {
            if (material_cache.count(mesh->material)) {
                mat = material_cache[mesh->material];
            } else {
                mat = convertMaterial(mesh->material);
                material_cache[mesh->material] = mat;
            }
        }
        
        Vector3f emission(0.0f, 0.0f, 0.0f);
        if (mesh->areaLight) {
            if (auto light = std::dynamic_pointer_cast<pbrt::DiffuseAreaLightRGB>(mesh->areaLight)) {
                emission = toVector3f(light->L);
            }
        }
        
        int num_faces = mesh->index.size();
        bool has_normals = mesh->normal.size() >= mesh->vertex.size();
        
        for (int i = 0; i < num_faces; i++) {
            const pbrt::vec3i& idx = mesh->index[i];
            
            Vector3f v0 = toVector3f(mesh->vertex[idx.x]);
            Vector3f v1 = toVector3f(mesh->vertex[idx.y]);
            Vector3f v2 = toVector3f(mesh->vertex[idx.z]);
            
            Vector3f normal;
            if (has_normals) {
                normal = unit_vector(toVector3f(mesh->normal[idx.x]));
            } else {
                normal = unit_vector(cross(v1 - v0, v2 - v0));
            }
            
            Triangle tri(v0, v1, v2, mat.getBSDF(), normal);
            tri.Le = emission;
            
            temp_primitives.push_back(Primitive(tri));
            num_triangles++;
        }
    }
    
    if (temp_primitives.empty()) {
        std::cerr << "Error: No triangles found in PBRT scene" << std::endl;
        return false;
    }
    
    // Copy to output
    num_primitives_out = static_cast<int>(temp_primitives.size());
    *h_list_out = new Primitive[num_primitives_out];
    
    for (int i = 0; i < num_primitives_out; i++) {
        (*h_list_out)[i] = temp_primitives[i];
    }
    
    std::cout << "========== PBRT SCENE LOADED ==========" << std::endl;
    std::cout << "  Meshes: " << num_meshes << std::endl;
    std::cout << "  Triangles: " << num_triangles << std::endl;
    std::cout << "  Materials: " << material_cache.size() << std::endl;
    std::cout << "========================================" << std::endl;
    
    return true;
}

#endif // PBRT_LOADER_H
