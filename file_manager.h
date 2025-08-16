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

#include "vec3.h"
#include "triangle.h"

// By marking all functions in this header as 'static', we are telling the compiler
// to create a private copy for each .cpp file that includes it. This completely
// avoids the "multiple definition" linker errors that normally happen when you
// put function bodies in a header file.

// A simple struct to hold material properties read from the MTL file
struct Material {
    Vector3f bsdf = Vector3f(0.8f, 0.8f, 0.8f); // Default to white
    Vector3f Le   = Vector3f(0.0f, 0.0f, 0.0f);   // Default to no emission
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


// Marked static to keep it private to this file's scope.
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
        } else if (prefix == "Kd") { // Diffuse color
            float r, g, b;
            iss >> r >> g >> b;
            current_material.bsdf = Vector3f(r, g, b);
        } else if (prefix == "Ke") { // Emissive color
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

static bool loadOBJ(const std::string& obj_filename, Triangle** h_list_out, int& num_triangles_out) {
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
    std::vector<Triangle> temp_triangles;

    std::map<std::string, Material> materials;
    Material current_material;

    std::string line;
    int line_num = 0;
    while (std::getline(file, line)) {
        line_num++;
        if (line.empty() || line[0] == '#') continue;

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
        } else if (prefix == "vn") {
            float nx, ny, nz;
            if (!(iss >> nx >> ny >> nz)) {
                 std::cerr << "Warning on line " << line_num << ": Malformed normal data." << std::endl;
                continue;
            }
            normals.emplace_back(unit_vector(Vector3f(nx, ny, nz)));
        } else if (prefix == "mtllib") {
            std::string mtl_filename;
            iss >> mtl_filename;
            materials = loadMTL(path_base + mtl_filename);
        } else if (prefix == "usemtl") {
            std::string material_name;
            iss >> material_name;
            if (materials.count(material_name)) {
                current_material = materials[material_name];
            } else {
                std::cerr << "Warning: Material '" << material_name << "' not found. Using default." << std::endl;
                current_material = Material();
            }
        } else if (prefix == "f") {
            std::vector<size_t> v_idx(3), n_idx(3);

            for (int i = 0; i < 3; ++i) {
                std::string token;
                iss >> token;  // Example: "2//1" or "2/5/1" or "2/5" or "2"

                size_t v = 0, vt = 0, vn = 0;
                char slash;

                std::stringstream tokenStream(token);

                if (!(tokenStream >> v)) {
                    std::cerr << "Warning on line " << line_num << ": malformed face vertex token '" << token << "'.\n";
                    v = 0;
                }

                if (tokenStream.peek() == '/') {
                    tokenStream >> slash; // consume first slash

                    if (tokenStream.peek() == '/') {
                        tokenStream >> slash; // consume second slash
                        tokenStream >> vn;    // v//vn format
                    } else {
                        tokenStream >> vt; // read vt
                        if (tokenStream.peek() == '/') {
                            tokenStream >> slash;
                            tokenStream >> vn; // v/vt/vn format
                        }
                    }
                }

                v_idx[i] = v;
                n_idx[i] = vn;
            }

            // Validate vertex indices
            if (v_idx[0] == 0 || v_idx[1] == 0 || v_idx[2] == 0 ||
                v_idx[0] > vertices.size() || v_idx[1] > vertices.size() || v_idx[2] > vertices.size()) {
                std::cerr << "Warning on line " << line_num << ": invalid vertex index.\n";
                continue;
            }

            // Retrieve vertices
            Vector3f vA = vertices[v_idx[0] - 1];
            Vector3f vB = vertices[v_idx[1] - 1];
            Vector3f vC = vertices[v_idx[2] - 1];

            // Determine normal
            Vector3f tri_normal;
            if (n_idx[0] != 0 && n_idx[0] <= normals.size()) {
                tri_normal = normals[n_idx[0] - 1];
            } else {
                tri_normal = unit_vector(cross(vB - vA, vC - vA));
            }

            Triangle tri(vA, vB, vC, current_material.bsdf, tri_normal);
            tri.Le = current_material.Le;
            temp_triangles.push_back(tri);
        }

    }
    file.close();

    if (temp_triangles.empty()) {
        std::cerr << "Error: No valid triangles were loaded from " << obj_filename << std::endl;
        return false;
    }

    num_triangles_out = static_cast<int>(temp_triangles.size());
    checkCudaErrors(cudaMallocHost(h_list_out, num_triangles_out * sizeof(Triangle)));
    memcpy(*h_list_out, temp_triangles.data(), num_triangles_out * sizeof(Triangle));

    std::cout << "Successfully loaded " << num_triangles_out << " triangles from " << obj_filename << std::endl;
    return true;
};

#endif // FILE_MANAGER_H