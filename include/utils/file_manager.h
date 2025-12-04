#ifndef FILE_MANAGER_H
#define FILE_MANAGER_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cuda_runtime.h>

// filesystem is C++17 and host-only
#ifndef __CUDA_ARCH__
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include "core/vector.h"
#include "rendering/triangle.h"
#include "rendering/quad.h"
#include "rendering/bvh.h"

struct Material {
    Vector3f bsdf = Vector3f(0.8f, 0.8f, 0.8f);
    Vector3f Le   = Vector3f(0.0f, 0.0f, 0.0f);
};

static void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

static std::map<std::string, Material> loadMTL(const std::string& filename) {
    std::map<std::string, Material> materials;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open MTL file: " << filename << std::endl;
        return materials;
    }

    std::string current_material_name;
    Material current_material;

    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "newmtl") {
            if (!current_material_name.empty()) {
                materials[current_material_name] = current_material;
            }
            iss >> current_material_name;
            current_material = Material();
        } else if (prefix == "Kd") {
            float r, g, b;
            iss >> r >> g >> b;
            current_material.bsdf = Vector3f(r, g, b);
        } else if (prefix == "Ke") {
            float r, g, b;
            iss >> r >> g >> b;
            current_material.Le = Vector3f(r, g, b);
        }
    }

    if (!current_material_name.empty()) {
        materials[current_material_name] = current_material;
    }

    std::cout << "Loaded " << materials.size() << " materials from " << filename << std::endl;
    return materials;
}

// Helper: Group triangles into quads if they share an edge
static bool tryMergeTrianglesIntoQuad(const std::vector<Triangle>& triangles, 
                                      std::vector<Primitive>& primitives) {
    // Simple heuristic: group consecutive triangle pairs that might be quads
    // This is optional - mainly for when loading triangle-only OBJ files
    for (size_t i = 0; i < triangles.size(); i++) {
        primitives.push_back(Primitive(triangles[i]));
    }
    return true;
}

// UNIFIED ROBUST OBJ LOADER with normal support
static bool loadOBJ(const std::string& obj_filename, Primitive** h_list_out, int& num_primitives_out) {
    std::ifstream file(obj_filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open OBJ file: " << obj_filename << "\n";
        return false;
    }

    std::string path_base = "";
    size_t last_slash = obj_filename.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        path_base = obj_filename.substr(0, last_slash + 1);
    }
    
    std::vector<Vector3f> vertices;
    std::vector<Vector3f> normals;
    std::vector<Primitive> temp_primitives;

    std::map<std::string, Material> materials;
    Material current_material;

    std::string line;
    int line_num = 0;
    int num_triangles = 0;
    int num_quads = 0;
    
    while (std::getline(file, line)) {
        line_num++;
        if (line.empty() || line[0] == '#' || line[0] == 'o' || line[0] == 's') continue;

        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            float x, y, z;
            if (!(iss >> x >> y >> z)) {
                std::cerr << "Warning on line " << line_num << ": Malformed vertex data." << std::endl;
                continue;
            }
            vertices.emplace_back(x, y, z);
        } 
        else if (prefix == "vn") {
            float nx, ny, nz;
            if (!(iss >> nx >> ny >> nz)) {
                std::cerr << "Warning on line " << line_num << ": Malformed normal data." << std::endl;
                continue;
            }
            normals.emplace_back(unit_vector(Vector3f(nx, ny, nz)));
        } 
        else if (prefix == "mtllib") {
            std::string mtl_filename;
            iss >> mtl_filename;
            materials = loadMTL(path_base + mtl_filename);
        } 
        else if (prefix == "usemtl") {
            std::string material_name;
            iss >> material_name;
            if (materials.count(material_name)) {
                current_material = materials[material_name];
            } else {
                std::cerr << "Warning: Material '" << material_name << "' not found. Using default." << std::endl;
                current_material = Material();
            }
        } 
        else if (prefix == "f") {
            std::vector<size_t> v_indices;
            std::vector<size_t> n_indices;
            std::string token;
            
            while (iss >> token) {
                size_t v = 0, vt = 0, vn = 0;
                char slash;
                std::stringstream tokenStream(token);
                
                if (!(tokenStream >> v)) {
                    std::cerr << "Warning on line " << line_num << ": malformed face vertex token '" << token << "'.\n";
                    continue;
                }
                
                // Parse v/vt/vn or v//vn or v/vt or v
                if (tokenStream.peek() == '/') {
                    tokenStream >> slash;
                    if (tokenStream.peek() == '/') {
                        tokenStream >> slash;
                        if (tokenStream >> vn) {}
                    } else {
                        if (tokenStream >> vt) {
                            if (tokenStream.peek() == '/') {
                                tokenStream >> slash;
                                if (tokenStream >> vn) {}
                            }
                        }
                    }
                }
                
                v_indices.push_back(v);
                n_indices.push_back(vn);
            }
            
            // Process triangle (3 vertices)
            if (v_indices.size() == 3) {
                if (v_indices[0] == 0 || v_indices[1] == 0 || v_indices[2] == 0 ||
                    v_indices[0] > vertices.size() || v_indices[1] > vertices.size() || v_indices[2] > vertices.size()) {
                    std::cerr << "Warning on line " << line_num << ": invalid vertex index.\n";
                    continue;
                }
                
                Vector3f v0 = vertices[v_indices[0] - 1];
                Vector3f v1 = vertices[v_indices[1] - 1];
                Vector3f v2 = vertices[v_indices[2] - 1];
                
                // Use provided normal if available, otherwise compute
                Vector3f tri_normal;
                if (n_indices[0] != 0 && n_indices[0] <= normals.size()) {
                    tri_normal = normals[n_indices[0] - 1];
                } else {
                    tri_normal = unit_vector(cross(v1 - v0, v2 - v0));
                }
                
                Triangle tri(v0, v1, v2, current_material.bsdf, tri_normal);
                tri.Le = current_material.Le;
                
                temp_primitives.push_back(Primitive(tri));
                num_triangles++;
            }
            // Process quad (4 vertices)
            else if (v_indices.size() == 4) {
                if (v_indices[0] == 0 || v_indices[1] == 0 || v_indices[2] == 0 || v_indices[3] == 0 ||
                    v_indices[0] > vertices.size() || v_indices[1] > vertices.size() || 
                    v_indices[2] > vertices.size() || v_indices[3] > vertices.size()) {
                    std::cerr << "Warning on line " << line_num << ": invalid vertex index.\n";
                    continue;
                }
                
                Vector3f v0 = vertices[v_indices[0] - 1];
                Vector3f v1 = vertices[v_indices[1] - 1];
                Vector3f v2 = vertices[v_indices[2] - 1];
                Vector3f v3 = vertices[v_indices[3] - 1];
                
                Quad quad(v0, v1, v2, v3, current_material.bsdf);
                
                // Override computed normal with OBJ normal if available
                if (n_indices[0] != 0 && n_indices[0] <= normals.size()) {
                    quad.normal = normals[n_indices[0] - 1];
                }
                
                quad.Le = current_material.Le;
                
                temp_primitives.push_back(Primitive(quad));
                num_quads++;
            }
            else {
                std::cerr << "Warning on line " << line_num << ": face with " 
                         << v_indices.size() << " vertices not supported.\n";
            }
        }
    }
    file.close();

    if (temp_primitives.empty()) {
        std::cerr << "Error: No valid primitives were loaded from " << obj_filename << std::endl;
        return false;
    }

    num_primitives_out = static_cast<int>(temp_primitives.size());
    *h_list_out = new Primitive[num_primitives_out];
    
    for (int i = 0; i < num_primitives_out; i++) {
        (*h_list_out)[i] = temp_primitives[i];
    }

    std::cout << "Successfully loaded " << num_primitives_out << " primitives from " << obj_filename << std::endl;
    if (num_triangles > 0) {
        std::cout << "  - " << num_triangles << " triangles" << std::endl;
    }
    if (num_quads > 0) {
        std::cout << "  - " << num_quads << " quads" << std::endl;
    }
    
    return true;
}

// ============================================================================
// UNIFIED SCENE LOADER - Handles OBJ and PBRT files
// ============================================================================

// PBRT Loader and loadScene are host-only
#ifndef __CUDA_ARCH__

// Forward declaration - defined in pbrt_loader.h when PBRT support is enabled
#ifdef USE_PBRT_LOADER
#include "utils/pbrt_loader.h"
#endif

/**
 * Unified scene loader that automatically detects file format
 * Supports: .obj, .pbrt
 */
static bool loadScene(const std::string& filename, Primitive** h_list_out, int& num_primitives_out) {
    fs::path filepath(filename);
    std::string extension = filepath.extension().string();
    
    // Convert to lowercase for comparison
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension == ".obj") {
        return loadOBJ(filename, h_list_out, num_primitives_out);
    }
#ifdef USE_PBRT_LOADER
    else if (extension == ".pbrt") {
        return loadPBRT(filename, h_list_out, num_primitives_out);
    }
#endif
    else {
        std::cerr << "Error: Unsupported file format '" << extension << "'" << std::endl;
        std::cerr << "Supported formats: .obj";
#ifdef USE_PBRT_LOADER
        std::cerr << ", .pbrt";
#endif
        std::cerr << std::endl;
        return false;
    }
}

#endif // !__CUDA_ARCH__

#endif // FILE_MANAGER_H
