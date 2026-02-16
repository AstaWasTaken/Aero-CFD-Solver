#pragma once

#include "cfd_core/mesh.hpp"
#include "cfd_core/numerics/euler_flux.hpp"

#include <array>
#include <string>
#include <vector>

namespace cfd::cuda_backend {
struct EulerResidualConfig {
  float gamma = 1.4f;
  bool use_second_order = false;
  cfd::core::ConservativeState farfield_state;
};

struct EulerDeviceBuffers {
  int num_cells = 0;
  int num_faces = 0;

  int* d_face_owner = nullptr;
  int* d_face_neighbor = nullptr;
  int* d_face_bc_type = nullptr;

  float* d_face_normal = nullptr;
  float* d_face_center = nullptr;
  float* d_face_area = nullptr;
  float* d_cell_center = nullptr;

  float* d_conserved = nullptr;
  float* d_gradients = nullptr;
  float* d_residual = nullptr;
  float* d_spectral_radius = nullptr;
};

bool cuda_runtime_available(std::string* error_message = nullptr);
bool cuda_euler_available(std::string* error_message = nullptr);

bool compute_scalar_residual_cuda(const cfd::core::UnstructuredMesh& mesh,
                                  const std::vector<float>& phi,
                                  const std::array<float, 3>& u_inf, float inflow_phi,
                                  std::vector<float>* residual, std::string* error_message);

bool init_euler_device_buffers(const cfd::core::UnstructuredMesh& mesh,
                               const std::vector<int>& face_bc_type,
                               EulerDeviceBuffers* buffers, std::string* error_message);
void free_euler_device_buffers(EulerDeviceBuffers* buffers);

bool euler_residual_cuda(const cfd::core::UnstructuredMesh& mesh,
                         const EulerResidualConfig& config, EulerDeviceBuffers* buffers,
                         const float* conserved, const float* gradients, float* residual,
                         float* spectral_radius, std::string* error_message);
}  // namespace cfd::cuda_backend
