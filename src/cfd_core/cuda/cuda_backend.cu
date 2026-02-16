#include "cfd_core/cuda_backend.hpp"
#include "kernels.cuh"

#include <cuda_runtime.h>

#include <string>
#include <vector>

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

bool compute_scalar_residual_cuda(const cfd::core::UnstructuredMesh& mesh,
                                  const std::vector<float>& phi,
                                  const std::array<float, 3>& u_inf, const float inflow_phi,
                                  std::vector<float>* residual, std::string* error_message) {
  if (mesh.num_cells <= 0 || mesh.num_faces <= 0 || residual == nullptr) {
    if (error_message != nullptr) {
      *error_message = "invalid mesh or output container";
    }
    return false;
  }
  if (phi.size() != static_cast<std::size_t>(mesh.num_cells)) {
    if (error_message != nullptr) {
      *error_message = "phi size does not match mesh.num_cells";
    }
    return false;
  }
  if (!cuda_runtime_available(error_message)) {
    return false;
  }

  int* d_face_owner = nullptr;
  int* d_face_neighbor = nullptr;
  float* d_face_normal = nullptr;
  float* d_face_area = nullptr;
  float* d_phi = nullptr;
  float* d_residual = nullptr;

  auto cleanup = [&]() {
    cudaFree(d_face_owner);
    cudaFree(d_face_neighbor);
    cudaFree(d_face_normal);
    cudaFree(d_face_area);
    cudaFree(d_phi);
    cudaFree(d_residual);
  };

  if (!check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_face_owner),
                             sizeof(int) * mesh.face_owner.size()),
                  "cudaMalloc(face_owner)", error_message) ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_face_neighbor),
                             sizeof(int) * mesh.face_neighbor.size()),
                  "cudaMalloc(face_neighbor)", error_message) ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_face_normal),
                             sizeof(float) * mesh.face_normal.size()),
                  "cudaMalloc(face_normal)", error_message) ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_face_area),
                             sizeof(float) * mesh.face_area.size()),
                  "cudaMalloc(face_area)", error_message) ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_phi), sizeof(float) * phi.size()),
                  "cudaMalloc(phi)", error_message) ||
      !check_cuda(cudaMalloc(reinterpret_cast<void**>(&d_residual), sizeof(float) * mesh.num_cells),
                  "cudaMalloc(residual)", error_message)) {
    cleanup();
    return false;
  }

  if (!check_cuda(cudaMemcpy(d_face_owner, mesh.face_owner.data(),
                             sizeof(int) * mesh.face_owner.size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy(face_owner)", error_message) ||
      !check_cuda(cudaMemcpy(d_face_neighbor, mesh.face_neighbor.data(),
                             sizeof(int) * mesh.face_neighbor.size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy(face_neighbor)", error_message) ||
      !check_cuda(cudaMemcpy(d_face_normal, mesh.face_normal.data(),
                             sizeof(float) * mesh.face_normal.size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy(face_normal)", error_message) ||
      !check_cuda(cudaMemcpy(d_face_area, mesh.face_area.data(),
                             sizeof(float) * mesh.face_area.size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy(face_area)", error_message) ||
      !check_cuda(cudaMemcpy(d_phi, phi.data(), sizeof(float) * phi.size(), cudaMemcpyHostToDevice),
                  "cudaMemcpy(phi)", error_message) ||
      !check_cuda(cudaMemset(d_residual, 0, sizeof(float) * mesh.num_cells), "cudaMemset(residual)",
                  error_message)) {
    cleanup();
    return false;
  }

  if (!launch_scalar_face_kernel(mesh.num_faces, d_face_owner, d_face_neighbor, d_face_normal,
                                 d_face_area, d_phi, inflow_phi, u_inf[0], u_inf[1], u_inf[2],
                                 d_residual, error_message)) {
    cleanup();
    return false;
  }

  residual->assign(static_cast<std::size_t>(mesh.num_cells), 0.0f);
  if (!check_cuda(cudaMemcpy(residual->data(), d_residual, sizeof(float) * mesh.num_cells,
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy(residual)", error_message)) {
    cleanup();
    return false;
  }

  cleanup();
  return true;
}
}  // namespace cfd::cuda_backend
