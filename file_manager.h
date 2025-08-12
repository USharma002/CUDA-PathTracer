#ifndef FILE_MANAGER_H
#define FILE_MANAGER_H

#include <map> // Add this include for std::map

#include "vec3.h"
#include "triangle.h"

vec3 computeNormal(const vec3& a, const vec3& b, const vec3& c) {
    return unit_vector(cross(b - a, c - a));
}

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// A simple struct to hold material properties read from the MTL file
struct Material {
    vec3 bsdf = vec3(0.8f, 0.8f, 0.8f); // Default to white
    vec3 Le = vec3(0.0f, 0.0f, 0.0f);   // Default to no emission
};
// Parses a .mtl file and returns a map of material names to Material properties.
std::map<std::string, Material> loadMTL(const std::string& filename) {
    std::map<std::string, Material> materials;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open MTL file: " << filename << std::endl;
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
            // If we were already building a material, save it before starting a new one.
            if (!current_material_name.empty()) {
                materials[current_material_name] = current_material;
            }
            iss >> current_material_name;
            current_material = Material(); // Reset for the new material
        } else if (prefix == "Kd") { // Diffuse color
            float r, g, b;
            iss >> r >> g >> b;
            current_material.bsdf = vec3(r, g, b);
        } else if (prefix == "Ke") { // Emissive color
            float r, g, b;
            iss >> r >> g >> b;
            current_material.Le = vec3(r, g, b);
        }
    }

    // Save the last material in the file
    if (!current_material_name.empty()) {
        materials[current_material_name] = current_material;
    }

    return materials;
}

// An updated OBJ loader that also parses material files (.mtl)
bool loadOBJ(const std::string& obj_filename, triangle** h_list_out, int& num_triangles_out) {
    std::ifstream file(obj_filename);
    if (!file.is_open()) {
        std::cerr << "Cannot open OBJ file: " << obj_filename << "\n";
        return false;
    }

    // Extract path from filename to find the MTL file
    std::string path_base = "";
    size_t last_slash = obj_filename.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        path_base = obj_filename.substr(0, last_slash + 1);
    }
    
    std::vector<vec3> vertices;
    std::vector<vec3> normals;
    std::vector<triangle> temp_triangles;

    std::map<std::string, Material> materials;
    Material current_material;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            float x, y, z;
            iss >> x >> y >> z;
            vertices.emplace_back(x, y, z);
        } else if (prefix == "vn") {
            float nx, ny, nz;
            iss >> nx >> ny >> nz;
            normals.emplace_back(nx, ny, nz);
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
            std::string face_data = line.substr(line.find(" ") + 1);
            std::replace(face_data.begin(), face_data.end(), '/', ' ');
            std::istringstream fss(face_data);

            int v_idx[3], n_idx[3] = {-1, -1, -1};
            int dummy_vt; // To consume texture coord indices if they exist

            for(int i = 0; i < 3; ++i) {
                fss >> v_idx[i];
                if (fss.peek() == ' ') {
                    fss.ignore();
                    if(fss.peek() == ' '){ // Format is v//n
                        fss.ignore();
                        fss >> n_idx[i];
                    } else { // Format is v/vt or v/vt/n
                        fss >> dummy_vt;
                        if(fss.peek() == ' '){
                            fss.ignore();
                            fss >> n_idx[i];
                        }
                    }
                }
            }

            vec3 vA = vertices[v_idx[0] - 1];
            vec3 vB = vertices[v_idx[1] - 1];
            vec3 vC = vertices[v_idx[2] - 1];

            vec3 tri_normal;
            if (n_idx[0] != -1) {
                // Average vertex normals for a smoother look if provided, though face normal is fine.
                // Here we just use the first vertex normal for the whole face.
                tri_normal = unit_vector(normals[n_idx[0] - 1]);
            } else {
                tri_normal = computeNormal(vA, vB, vC); // Compute face normal if not provided
            }

            triangle tri(vA, vB, vC, current_material.bsdf, tri_normal);
            tri.Le = current_material.Le;
            temp_triangles.push_back(tri);
        }
    }
    file.close();

    num_triangles_out = static_cast<int>(temp_triangles.size());
    checkCudaErrors(cudaMallocHost(h_list_out, num_triangles_out * sizeof(triangle)));

    // Copy the loaded triangles into the pinned host memory
    memcpy(*h_list_out, temp_triangles.data(), num_triangles_out * sizeof(triangle));

    std::cout << "Loaded " << num_triangles_out << " triangles from " << obj_filename << std::endl;

    return true;
};

#endif  // FILE_MANAGER_H