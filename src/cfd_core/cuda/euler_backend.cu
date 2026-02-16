#include "cfd_core/cuda_backend.hpp"

#include "euler_kernels.cuh"

#include <cuda_runtime.h>

#include <string>

namespace cfd::cuda_backend {
namespace {
bool check_cuda(const cudaError_t status, const std::string& context, std::string* error_message) {
  if (status == cudaSuccess) {
    return true;
  }
  if (error_message != nullptr) {
    *error_message = context + ": " + cudaGetErrorString(status);
  }
  return false;
}
}  // namespace

bool cuda_runtime_available(std::string* error_message) {
  int device_count = 0;
  const cudaError_t status = cudaGetDeviceCount(&device_count);
  if (!check_cuda(status, "cudaGetDeviceCount", error_message)) {
    return false;
  }
  if (device_count <= 0) {
    if (error_message != nullptr) {
      *error_message = "No CUDA device detected.";
    }
    return false;
  }
  return true;
}

bool cuda_euler_available(std::string* error_message) {
  return cuda_runtime_available(error_message);
}

void free_euler_device_buffers(EulerDeviceBuffers* buffers) {
  if (buffers == nullptr) {
    return;
  }

  cudaFree(buffers->d_face_owner);
  cudaFree(buffers->d_face_neighbor);
  cudaFree(buffers->d_face_bc_type);
  cudaFree(buffers->d_face_normal);
  cudaFree(buffers->d_face_center);
  cudaFree(buffers->d_face_area);
  cudaFree(buffers->d_cell_center);
  cudaFree(buffers->d_conserved);
  cudaFree(buffers->d_gradients);
  cudaFree(buffers->d_residual);
  cudaFree(buffers->d_spectral_radius);

  *buffers = EulerDeviceBuffers();
}

bool init_euler_device_buffers(const cfd::core::UnstructuredMesh& mesh,
                               const std::vector<int>& face_bc_type,
                               EulerDeviceBuffers* buffers, std::string* error_message) {
  if (buffers == nullptr) {
    if (error_message != nullptr) {
      *error_message = "EulerDeviceBuffers output is null.";
    }
    return false;
  }
  if (!cuda_runtime_available(error_message)) {
    return false;
  }
  if (mesh.num_cells <= 0 || mesh.num_faces <= 0) {
    if (error_message != nullptr) {
      *error_message = "Mesh must contain cells and faces.";
    }
    return false;
  }
  if (mesh.face_owner.size() != static_cast<std::size_t>(mesh.num_faces) ||
      mesh.face_neighbor.size() != static_cast<std::size_t>(mesh.num_faces) ||
      mesh.face_normal.size() != static_cast<std::size_t>(mesh.num_faces) * 3 ||
      mesh.face_center.size() != static_cast<std::size_t>(mesh.num_faces) * 3 ||
      mesh.face_area.size() != static_cast<std::size_t>(mesh.num_faces) ||
      mesh.cell_center.size() != static_cast<std::size_t>(mesh.num_cells) * 3 ||
      face_bc_type.size() != static_cast<std::size_t>(mesh.num_faces)) {
    if (error_message != nullptr) {
      *error_message = "Mesh arrays or boundary labels have inconsistent sizes.";
    }
    return false;
  }

  free_euler_device_buffers(buffers);
  buffers->num_cells = mesh.num_cells;
  buffers->num_faces = mesh.num_faces;

  auto cleanup = [&]() { free_euler_device_buffers(buffers); };

  if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers->d_face_owner),
                             sizeof(int) * mesh.face_owner.size()),
                  "cudaMalloc(face_owner)", error_message) ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers->d_face_neighbor),
                             sizeof(int) * mesh.face_neighbor.size()),
                  "cudaMalloc(face_neighbor)", error_message) ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers->d_face_bc_type),
                             sizeof(int) * face_bc_type.size()),
                  "cudaMalloc(face_bc_type)", error_message) ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers->d_face_normal),
                             sizeof(float) * mesh.face_normal.size()),
                  "cudaMalloc(face_normal)", error_message) ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers->d_face_center),
                             sizeof(float) * mesh.face_center.size()),
                  "cudaMalloc(face_center)", error_message) ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers->d_face_area),
                             sizeof(float) * mesh.face_area.size()),
                  "cudaMalloc(face_area)", error_message) ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers->d_cell_center),
                             sizeof(float) * mesh.cell_center.size()),
                  "cudaMalloc(cell_center)", error_message) ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers->d_conserved),
                             sizeof(float) * static_cast<std::size_t>(mesh.num_cells) * 5),
                  "cudaMalloc(conserved)", error_message) ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers->d_gradients),
                             sizeof(float) * static_cast<std::size_t>(mesh.num_cells) * 5 * 2),
                  "cudaMalloc(gradients)", error_message) ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers->d_residual),
                             sizeof(float) * static_cast<std::size_t>(mesh.num_cells) * 5),
                  "cudaMalloc(residual)", error_message) ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&buffers->d_spectral_radius),
                             sizeof(float) * static_cast<std::size_t>(mesh.num_cells)),
                  "cudaMalloc(spectral_radius)", error_message)) {
    cleanup();
    return false;
  }

  if (!check_cuda(cudaMemcpy(buffers->d_face_owner, mesh.face_owner.data(),
                             sizeof(int) * mesh.face_owner.size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy(face_owner)", error_message) ||
      !check_cuda(cudaMemcpy(buffers->d_face_neighbor, mesh.face_neighbor.data(),
                             sizeof(int) * mesh.face_neighbor.size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy(face_neighbor)", error_message) ||
      !check_cuda(cudaMemcpy(buffers->d_face_bc_type, face_bc_type.data(),
                             sizeof(int) * face_bc_type.size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy(face_bc_type)", error_message) ||
      !check_cuda(cudaMemcpy(buffers->d_face_normal, mesh.face_normal.data(),
                             sizeof(float) * mesh.face_normal.size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy(face_normal)", error_message) ||
      !check_cuda(cudaMemcpy(buffers->d_face_center, mesh.face_center.data(),
                             sizeof(float) * mesh.face_center.size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy(face_center)", error_message) ||
      !check_cuda(cudaMemcpy(buffers->d_face_area, mesh.face_area.data(),
                             sizeof(float) * mesh.face_area.size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy(face_area)", error_message) ||
      !check_cuda(cudaMemcpy(buffers->d_cell_center, mesh.cell_center.data(),
                             sizeof(float) * mesh.cell_center.size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy(cell_center)", error_message)) {
    cleanup();
    return false;
  }

  return true;
}

bool euler_residual_cuda(const cfd::core::UnstructuredMesh& mesh,
                         const EulerResidualConfig& config, EulerDeviceBuffers* buffers,
                         const float* conserved, const float* gradients, float* residual,
                         float* spectral_radius, std::string* error_message) {
  if (buffers == nullptr || conserved == nullptr || residual == nullptr || spectral_radius == nullptr) {
    if (error_message != nullptr) {
      *error_message = "Null pointer passed to euler_residual_cuda.";
    }
    return false;
  }
  if (!cuda_runtime_available(error_message)) {
    return false;
  }
  if (buffers->num_cells != mesh.num_cells || buffers->num_faces != mesh.num_faces) {
    if (error_message != nullptr) {
      *error_message = "Device buffer sizes do not match mesh.";
    }
    return false;
  }

  const std::size_t num_conserved = static_cast<std::size_t>(mesh.num_cells) * 5;
  const std::size_t num_gradients = static_cast<std::size_t>(mesh.num_cells) * 5 * 2;
  const std::size_t num_residual = static_cast<std::size_t>(mesh.num_cells) * 5;
  const std::size_t num_spectral = static_cast<std::size_t>(mesh.num_cells);

  if (!check_cuda(cudaMemcpy(buffers->d_conserved, conserved, sizeof(float) * num_conserved,
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy(conserved)", error_message)) {
    return false;
  }
  if (gradients != nullptr) {
    if (!check_cuda(cudaMemcpy(buffers->d_gradients, gradients, sizeof(float) * num_gradients,
                               cudaMemcpyHostToDevice),
                    "cudaMemcpy(gradients)", error_message)) {
      return false;
    }
  } else if (!check_cuda(cudaMemset(buffers->d_gradients, 0, sizeof(float) * num_gradients),
                         "cudaMemset(gradients)", error_message)) {
    return false;
  }
  if (!check_cuda(cudaMemset(buffers->d_residual, 0, sizeof(float) * num_residual),
                  "cudaMemset(residual)", error_message) ||
      !check_cuda(cudaMemset(buffers->d_spectral_radius, 0, sizeof(float) * num_spectral),
                  "cudaMemset(spectral_radius)", error_message)) {
    return false;
  }

  if (!launch_euler_face_kernel(
        mesh.num_faces, config.use_second_order ? 1 : 0, config.gamma, config.farfield_state.rho,
        config.farfield_state.rhou, config.farfield_state.rhov, config.farfield_state.rhow,
        config.farfield_state.rhoE, buffers->d_face_owner, buffers->d_face_neighbor,
        buffers->d_face_bc_type, buffers->d_face_normal, buffers->d_face_center,
        buffers->d_face_area, buffers->d_cell_center, buffers->d_conserved, buffers->d_gradients,
        buffers->d_residual, buffers->d_spectral_radius, error_message)) {
    return false;
  }

  if (!check_cuda(cudaMemcpy(residual, buffers->d_residual, sizeof(float) * num_residual,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(residual)", error_message) ||
      !check_cuda(cudaMemcpy(spectral_radius, buffers->d_spectral_radius,
                             sizeof(float) * num_spectral, cudaMemcpyDeviceToHost),
                  "cudaMemcpy(spectral_radius)", error_message)) {
    return false;
  }

  return true;
}
}  // namespace cfd::cuda_backend
