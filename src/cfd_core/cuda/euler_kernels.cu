#include "euler_kernels.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <string>

namespace {
constexpr int kNumVars = 5;
constexpr float kRhoFloor = 1.0e-8f;
constexpr float kPressureFloor = 1.0e-8f;
constexpr int kFarfieldBoundary = 1;
constexpr int kWallBoundary = 2;

struct PrimitiveStateDevice {
  float rho = 1.0f;
  float u = 0.0f;
  float v = 0.0f;
  float w = 0.0f;
  float p = 1.0f;
};

struct ConservativeStateDevice {
  float rho = 1.0f;
  float rhou = 0.0f;
  float rhov = 0.0f;
  float rhow = 0.0f;
  float rhoE = 1.0f;
};

__device__ ConservativeStateDevice load_conservative(const float* conserved, const int cell) {
  return {
    conserved[5 * cell + 0],
    conserved[5 * cell + 1],
    conserved[5 * cell + 2],
    conserved[5 * cell + 3],
    conserved[5 * cell + 4],
  };
}

__device__ float minmod(const float a, const float b) {
  if (a * b <= 0.0f) {
    return 0.0f;
  }
  return (fabsf(a) < fabsf(b)) ? a : b;
}

__device__ PrimitiveStateDevice conservative_to_primitive(const ConservativeStateDevice& conservative,
                                                          const float gamma) {
  PrimitiveStateDevice primitive;
  primitive.rho = fmaxf(conservative.rho, kRhoFloor);
  const float inv_rho = 1.0f / primitive.rho;
  primitive.u = conservative.rhou * inv_rho;
  primitive.v = conservative.rhov * inv_rho;
  primitive.w = conservative.rhow * inv_rho;
  const float kinetic =
    0.5f * primitive.rho *
    (primitive.u * primitive.u + primitive.v * primitive.v + primitive.w * primitive.w);
  primitive.p = fmaxf((gamma - 1.0f) * (conservative.rhoE - kinetic), kPressureFloor);
  return primitive;
}

__device__ ConservativeStateDevice primitive_to_conservative(const PrimitiveStateDevice& primitive,
                                                             const float gamma) {
  ConservativeStateDevice conservative;
  conservative.rho = fmaxf(primitive.rho, kRhoFloor);
  conservative.rhou = conservative.rho * primitive.u;
  conservative.rhov = conservative.rho * primitive.v;
  conservative.rhow = conservative.rho * primitive.w;
  const float kinetic = 0.5f * conservative.rho *
                        (primitive.u * primitive.u + primitive.v * primitive.v +
                         primitive.w * primitive.w);
  const float pressure = fmaxf(primitive.p, kPressureFloor);
  conservative.rhoE = pressure / (gamma - 1.0f) + kinetic;
  return conservative;
}

__device__ float speed_of_sound(const PrimitiveStateDevice& primitive, const float gamma) {
  return sqrtf(fmaxf(gamma * primitive.p / fmaxf(primitive.rho, kRhoFloor), 0.0f));
}

__device__ void rusanov_flux(const ConservativeStateDevice& left, const ConservativeStateDevice& right,
                             const float nx, const float ny, const float nz, const float gamma,
                             float* flux, float* max_wave_speed) {
  const PrimitiveStateDevice pl = conservative_to_primitive(left, gamma);
  const PrimitiveStateDevice pr = conservative_to_primitive(right, gamma);

  const float un_l = pl.u * nx + pl.v * ny + pl.w * nz;
  const float un_r = pr.u * nx + pr.v * ny + pr.w * nz;

  const float a_l = speed_of_sound(pl, gamma);
  const float a_r = speed_of_sound(pr, gamma);
  const float smax = fmaxf(fabsf(un_l) + a_l, fabsf(un_r) + a_r);
  if (max_wave_speed != nullptr) {
    *max_wave_speed = smax;
  }

  const float f_l[kNumVars] = {
    left.rho * un_l,
    left.rhou * un_l + nx * pl.p,
    left.rhov * un_l + ny * pl.p,
    left.rhow * un_l + nz * pl.p,
    (left.rhoE + pl.p) * un_l,
  };
  const float f_r[kNumVars] = {
    right.rho * un_r,
    right.rhou * un_r + nx * pr.p,
    right.rhov * un_r + ny * pr.p,
    right.rhow * un_r + nz * pr.p,
    (right.rhoE + pr.p) * un_r,
  };
  const float du[kNumVars] = {
    right.rho - left.rho,
    right.rhou - left.rhou,
    right.rhov - left.rhov,
    right.rhow - left.rhow,
    right.rhoE - left.rhoE,
  };
  for (int k = 0; k < kNumVars; ++k) {
    flux[k] = 0.5f * (f_l[k] + f_r[k]) - 0.5f * smax * du[k];
  }
}

__device__ void slip_wall_flux(const ConservativeStateDevice& interior, const float nx,
                               const float ny, const float nz, const float gamma, float* flux,
                               float* max_wave_speed) {
  const PrimitiveStateDevice primitive = conservative_to_primitive(interior, gamma);
  const float un = primitive.u * nx + primitive.v * ny + primitive.w * nz;
  const float a = speed_of_sound(primitive, gamma);
  if (max_wave_speed != nullptr) {
    *max_wave_speed = fabsf(un) + a;
  }

  flux[0] = 0.0f;
  flux[1] = primitive.p * nx;
  flux[2] = primitive.p * ny;
  flux[3] = primitive.p * nz;
  flux[4] = 0.0f;
}

__device__ float gradient_value(const float* gradients, const int cell, const int var,
                                const int component) {
  return gradients[(cell * kNumVars + var) * 2 + component];
}

__global__ void euler_face_kernel(const int num_faces, const int use_second_order,
                                  const float gamma, const float farfield_rho,
                                  const float farfield_rhou, const float farfield_rhov,
                                  const float farfield_rhow, const float farfield_rhoE,
                                  const int* face_owner, const int* face_neighbor,
                                  const int* face_bc_type, const float* face_normal,
                                  const float* face_center, const float* face_area,
                                  const float* cell_center, const float* conserved,
                                  const float* gradients, float* residual,
                                  float* spectral_radius) {
  const int face = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (face >= num_faces) {
    return;
  }

  const int owner = face_owner[face];
  const int neighbor = face_neighbor[face];

  const float nx = face_normal[3 * face + 0];
  const float ny = face_normal[3 * face + 1];
  const float nz = face_normal[3 * face + 2];
  const float area = face_area[face];

  float flux[kNumVars] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  float max_wave_speed = 0.0f;

  if (neighbor >= 0) {
    const ConservativeStateDevice owner_state = load_conservative(conserved, owner);
    const ConservativeStateDevice neighbor_state = load_conservative(conserved, neighbor);
    const PrimitiveStateDevice owner_primitive = conservative_to_primitive(owner_state, gamma);
    const PrimitiveStateDevice neighbor_primitive = conservative_to_primitive(neighbor_state, gamma);

    float ql[kNumVars] = {
      owner_primitive.rho,
      owner_primitive.u,
      owner_primitive.v,
      owner_primitive.w,
      owner_primitive.p,
    };
    float qr[kNumVars] = {
      neighbor_primitive.rho,
      neighbor_primitive.u,
      neighbor_primitive.v,
      neighbor_primitive.w,
      neighbor_primitive.p,
    };

    if (use_second_order != 0) {
      const float fx = face_center[3 * face + 0];
      const float fy = face_center[3 * face + 1];
      const float ox = cell_center[3 * owner + 0];
      const float oy = cell_center[3 * owner + 1];
      const float nx_cell = cell_center[3 * neighbor + 0];
      const float ny_cell = cell_center[3 * neighbor + 1];

      const float d_owner_x = fx - ox;
      const float d_owner_y = fy - oy;
      const float d_neighbor_x = fx - nx_cell;
      const float d_neighbor_y = fy - ny_cell;

      for (int var = 0; var < kNumVars; ++var) {
        const float qo = ql[var];
        const float qn = qr[var];
        const float grad_o_x = gradient_value(gradients, owner, var, 0);
        const float grad_o_y = gradient_value(gradients, owner, var, 1);
        const float grad_n_x = gradient_value(gradients, neighbor, var, 0);
        const float grad_n_y = gradient_value(gradients, neighbor, var, 1);

        const float delta_cell = qn - qo;
        const float delta_owner = grad_o_x * d_owner_x + grad_o_y * d_owner_y;
        const float delta_neighbor = grad_n_x * d_neighbor_x + grad_n_y * d_neighbor_y;

        ql[var] = qo + minmod(delta_owner, delta_cell);
        qr[var] = qn + minmod(delta_neighbor, -delta_cell);
      }
    }

    ql[0] = fmaxf(ql[0], kRhoFloor);
    ql[4] = fmaxf(ql[4], kPressureFloor);
    qr[0] = fmaxf(qr[0], kRhoFloor);
    qr[4] = fmaxf(qr[4], kPressureFloor);

    const PrimitiveStateDevice left_primitive = {ql[0], ql[1], ql[2], ql[3], ql[4]};
    const PrimitiveStateDevice right_primitive = {qr[0], qr[1], qr[2], qr[3], qr[4]};
    const ConservativeStateDevice left_state = primitive_to_conservative(left_primitive, gamma);
    const ConservativeStateDevice right_state = primitive_to_conservative(right_primitive, gamma);
    rusanov_flux(left_state, right_state, nx, ny, nz, gamma, flux, &max_wave_speed);
  } else {
    const ConservativeStateDevice owner_state = load_conservative(conserved, owner);
    const PrimitiveStateDevice owner_primitive = conservative_to_primitive(owner_state, gamma);
    const PrimitiveStateDevice left_primitive = {
      fmaxf(owner_primitive.rho, kRhoFloor),
      owner_primitive.u,
      owner_primitive.v,
      owner_primitive.w,
      fmaxf(owner_primitive.p, kPressureFloor),
    };
    const ConservativeStateDevice left_state = primitive_to_conservative(left_primitive, gamma);
    const int bc_type = face_bc_type[face];

    if (bc_type == kWallBoundary) {
      slip_wall_flux(left_state, nx, ny, nz, gamma, flux, &max_wave_speed);
    } else {
      const ConservativeStateDevice farfield_state = {
        farfield_rho,
        farfield_rhou,
        farfield_rhov,
        farfield_rhow,
        farfield_rhoE,
      };
      rusanov_flux(left_state, farfield_state, nx, ny, nz, gamma, flux, &max_wave_speed);
    }
  }

  const float wave = max_wave_speed * area;
  atomicAdd(&spectral_radius[owner], wave);
  for (int var = 0; var < kNumVars; ++var) {
    const float flux_area = flux[var] * area;
    atomicAdd(&residual[5 * owner + var], flux_area);
    if (neighbor >= 0) {
      atomicAdd(&residual[5 * neighbor + var], -flux_area);
    }
  }
  if (neighbor >= 0) {
    atomicAdd(&spectral_radius[neighbor], wave);
  }
}
}  // namespace

namespace cfd::cuda_backend {
bool launch_euler_face_kernel(const int num_faces, const int use_second_order, const float gamma,
                              const float farfield_rho, const float farfield_rhou,
                              const float farfield_rhov, const float farfield_rhow,
                              const float farfield_rhoE, const int* d_face_owner,
                              const int* d_face_neighbor, const int* d_face_bc_type,
                              const float* d_face_normal, const float* d_face_center,
                              const float* d_face_area, const float* d_cell_center,
                              const float* d_conserved, const float* d_gradients,
                              float* d_residual, float* d_spectral_radius,
                              std::string* error_message) {
  constexpr int threads_per_block = 256;
  const int blocks = (num_faces + threads_per_block - 1) / threads_per_block;
  euler_face_kernel<<<blocks, threads_per_block>>>(
    num_faces, use_second_order, gamma, farfield_rho, farfield_rhou, farfield_rhov,
    farfield_rhow, farfield_rhoE, d_face_owner, d_face_neighbor, d_face_bc_type, d_face_normal,
    d_face_center, d_face_area, d_cell_center, d_conserved, d_gradients, d_residual,
    d_spectral_radius);

  const cudaError_t launch_status = cudaGetLastError();
  if (launch_status != cudaSuccess) {
    if (error_message != nullptr) {
      *error_message = std::string("kernel launch failed: ") + cudaGetErrorString(launch_status);
    }
    return false;
  }

  const cudaError_t sync_status = cudaDeviceSynchronize();
  if (sync_status != cudaSuccess) {
    if (error_message != nullptr) {
      *error_message = std::string("kernel synchronize failed: ") + cudaGetErrorString(sync_status);
    }
    return false;
  }

  return true;
}
}  // namespace cfd::cuda_backend
