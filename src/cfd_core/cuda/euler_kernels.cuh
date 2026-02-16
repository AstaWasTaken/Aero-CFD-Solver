#pragma once

#include <string>

namespace cfd::cuda_backend {
bool launch_euler_face_kernel(int num_faces, int use_second_order, float gamma,
                              float farfield_rho, float farfield_rhou,
                              float farfield_rhov, float farfield_rhow,
                              float farfield_rhoE, const int* d_face_owner,
                              const int* d_face_neighbor, const int* d_face_bc_type,
                              const float* d_face_normal, const float* d_face_center,
                              const float* d_face_area, const float* d_cell_center,
                              const float* d_conserved, const float* d_gradients,
                              float* d_residual, float* d_spectral_radius,
                              std::string* error_message);
}  // namespace cfd::cuda_backend
