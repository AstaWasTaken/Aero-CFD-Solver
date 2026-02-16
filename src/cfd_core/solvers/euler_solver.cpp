#include "cfd_core/solvers/euler_solver.hpp"

#include "cfd_core/backend.hpp"
#include "cfd_core/io_vtk.hpp"
#include "cfd_core/numerics/euler_flux.hpp"
#include "cfd_core/numerics/reconstruction.hpp"
#include "cfd_core/post/forces.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#if CFD_HAS_CUDA
#include "cfd_core/cuda_backend.hpp"
#endif

namespace cfd::core {
namespace {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kRhoFloor = 1.0e-8f;
constexpr float kPressureFloor = 1.0e-8f;
constexpr int kFaceBoundaryInterior = 0;
constexpr int kFaceBoundaryFarfield = 1;
constexpr int kFaceBoundaryWall = 2;

ConservativeState load_conservative_state(const std::vector<float>& conserved, const int cell) {
  return {
    conserved[5 * cell + 0],
    conserved[5 * cell + 1],
    conserved[5 * cell + 2],
    conserved[5 * cell + 3],
    conserved[5 * cell + 4],
  };
}

void store_conservative_state(std::vector<float>* conserved, const int cell,
                              const ConservativeState& state) {
  if (conserved == nullptr) {
    return;
  }
  (*conserved)[5 * cell + 0] = state.rho;
  (*conserved)[5 * cell + 1] = state.rhou;
  (*conserved)[5 * cell + 2] = state.rhov;
  (*conserved)[5 * cell + 3] = state.rhow;
  (*conserved)[5 * cell + 4] = state.rhoE;
}

void enforce_physical_state(ConservativeState* state, const float gamma) {
  if (state == nullptr) {
    return;
  }
  state->rho = std::max(state->rho, kRhoFloor);
  const float inv_rho = 1.0f / state->rho;
  const float u = state->rhou * inv_rho;
  const float v = state->rhov * inv_rho;
  const float w = state->rhow * inv_rho;
  const float kinetic = 0.5f * state->rho * (u * u + v * v + w * w);
  state->rhoE = std::max(state->rhoE, kinetic + kPressureFloor / (gamma - 1.0f));
  const float p = (gamma - 1.0f) * (state->rhoE - kinetic);
  if (p < kPressureFloor) {
    state->rhoE = kinetic + kPressureFloor / (gamma - 1.0f);
  }
}

std::vector<std::string> build_face_patch_name(const UnstructuredMesh& mesh) {
  std::vector<std::string> patch_name(static_cast<std::size_t>(mesh.num_faces), std::string());
  for (const auto& patch : mesh.boundary_patches) {
    for (int i = 0; i < patch.face_count; ++i) {
      const int face = patch.start_face + i;
      patch_name[static_cast<std::size_t>(face)] = patch.name;
    }
  }
  return patch_name;
}

bool is_wall_patch(const std::string& patch_name) {
  return patch_name == "wall";
}

bool is_farfield_patch(const std::string& patch_name) {
  return patch_name == "farfield" || patch_name == "boundary";
}

std::vector<int> build_face_boundary_type(const UnstructuredMesh& mesh) {
  std::vector<int> face_boundary_type(static_cast<std::size_t>(mesh.num_faces), kFaceBoundaryInterior);
  const std::vector<std::string> face_patch_name = build_face_patch_name(mesh);
  for (int face = 0; face < mesh.num_faces; ++face) {
    if (mesh.face_neighbor[face] >= 0) {
      continue;
    }
    const std::string& patch = face_patch_name[static_cast<std::size_t>(face)];
    if (is_wall_patch(patch)) {
      face_boundary_type[static_cast<std::size_t>(face)] = kFaceBoundaryWall;
    } else if (is_farfield_patch(patch) || patch.empty()) {
      face_boundary_type[static_cast<std::size_t>(face)] = kFaceBoundaryFarfield;
    } else {
      face_boundary_type[static_cast<std::size_t>(face)] = kFaceBoundaryFarfield;
    }
  }
  return face_boundary_type;
}

float compute_cfl(const EulerAirfoilCaseConfig& config, const int iter) {
  if (config.cfl_ramp_iters <= 0) {
    return config.cfl_max;
  }
  const float t = std::min(1.0f, static_cast<float>(iter + 1) /
                                   static_cast<float>(std::max(config.cfl_ramp_iters, 1)));
  return config.cfl_start + t * (config.cfl_max - config.cfl_start);
}

std::array<float, 5> slip_wall_flux(const ConservativeState& interior,
                                    const std::array<float, 3>& unit_normal, const float gamma,
                                    float* max_wave_speed) {
  const PrimitiveState primitive = conservative_to_primitive(interior, gamma);
  const float un = primitive.u * unit_normal[0] + primitive.v * unit_normal[1] +
                   primitive.w * unit_normal[2];
  const float a = speed_of_sound(primitive, gamma);
  if (max_wave_speed != nullptr) {
    *max_wave_speed = std::abs(un) + a;
  }

  return {
    0.0f,
    primitive.p * unit_normal[0],
    primitive.p * unit_normal[1],
    primitive.p * unit_normal[2],
    0.0f,
  };
}

void assemble_euler_residual_cpu(const UnstructuredMesh& mesh,
                                 const std::vector<PrimitiveState>& primitive,
                                 const PrimitiveGradients& gradients,
                                 const std::vector<int>& face_boundary_type,
                                 const ConservativeState& conservative_inf, const float gamma,
                                 const bool use_second_order, std::vector<float>* residual,
                                 std::vector<float>* spectral_radius) {
  if (residual == nullptr || spectral_radius == nullptr) {
    return;
  }

  for (int face = 0; face < mesh.num_faces; ++face) {
    const int owner = mesh.face_owner[face];
    const int neighbor = mesh.face_neighbor[face];
    const std::array<float, 3> normal = {
      mesh.face_normal[3 * face + 0],
      mesh.face_normal[3 * face + 1],
      mesh.face_normal[3 * face + 2],
    };
    const float area = mesh.face_area[face];

    ConservativeState left_state;
    ConservativeState right_state;
    std::array<float, 5> flux = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float max_wave_speed = 0.0f;

    if (neighbor >= 0) {
      PrimitiveState left_primitive = primitive[static_cast<std::size_t>(owner)];
      PrimitiveState right_primitive = primitive[static_cast<std::size_t>(neighbor)];
      if (use_second_order) {
        reconstruct_interior_face_states(mesh, primitive, gradients, face, LimiterType::kMinmod,
                                         &left_primitive, &right_primitive);
      }
      left_state = primitive_to_conservative(left_primitive, gamma);
      right_state = primitive_to_conservative(right_primitive, gamma);
      flux = rusanov_flux(left_state, right_state, normal, gamma, &max_wave_speed);
    } else {
      const PrimitiveState owner_primitive = primitive[static_cast<std::size_t>(owner)];
      left_state = primitive_to_conservative(owner_primitive, gamma);
      const int bc_type = face_boundary_type[static_cast<std::size_t>(face)];
      if (bc_type == kFaceBoundaryWall) {
        flux = slip_wall_flux(left_state, normal, gamma, &max_wave_speed);
      } else {
        right_state = conservative_inf;
        flux = rusanov_flux(left_state, right_state, normal, gamma, &max_wave_speed);
      }
    }

    const float wave = max_wave_speed * area;
    (*spectral_radius)[static_cast<std::size_t>(owner)] += wave;
    for (int k = 0; k < 5; ++k) {
      const float flux_area = flux[k] * area;
      (*residual)[static_cast<std::size_t>(5 * owner + k)] += flux_area;
      if (neighbor >= 0) {
        (*residual)[static_cast<std::size_t>(5 * neighbor + k)] -= flux_area;
      }
    }

    if (neighbor >= 0) {
      (*spectral_radius)[static_cast<std::size_t>(neighbor)] += wave;
    }
  }
}

#if CFD_HAS_CUDA
struct EulerCudaBuffersGuard {
  cfd::cuda_backend::EulerDeviceBuffers buffers;

  ~EulerCudaBuffersGuard() {
    cfd::cuda_backend::free_euler_device_buffers(&buffers);
  }
};
#endif
}  // namespace

EulerRunResult run_euler_airfoil_case(const EulerAirfoilCaseConfig& config,
                                      const std::string& backend) {
  if (config.iterations <= 0) {
    throw std::invalid_argument("Euler iterations must be positive.");
  }
  const std::string resolved_backend = normalize_backend(backend);

  EulerRunResult result;
  result.mesh = make_airfoil_ogrid_mesh(config.mesh);

  const int num_cells = result.mesh.num_cells;
  const float gamma = config.gamma;

  float rho_inf = config.rho_inf;
  if (rho_inf <= 0.0f) {
    rho_inf = config.p_inf / (config.gas_constant * config.t_inf);
  }
  if (rho_inf <= 0.0f) {
    throw std::invalid_argument("Invalid freestream density. Set rho_inf or p_inf/T_inf.");
  }

  const float alpha = config.aoa_deg * (kPi / 180.0f);
  const float a_inf = std::sqrt(gamma * config.p_inf / rho_inf);
  const float v_inf = std::max(config.mach, 0.0f) * a_inf;
  const PrimitiveState primitive_inf = {
    rho_inf,
    v_inf * std::cos(alpha),
    v_inf * std::sin(alpha),
    0.0f,
    config.p_inf,
  };
  const ConservativeState conservative_inf = primitive_to_conservative(primitive_inf, gamma);

  result.conserved.assign(static_cast<std::size_t>(num_cells) * 5, 0.0f);
  for (int cell = 0; cell < num_cells; ++cell) {
    store_conservative_state(&result.conserved, cell, conservative_inf);
  }

  std::vector<PrimitiveState> primitive(static_cast<std::size_t>(num_cells));
  std::vector<float> residual(static_cast<std::size_t>(num_cells) * 5, 0.0f);
  std::vector<float> spectral_radius(static_cast<std::size_t>(num_cells), 0.0f);
  std::vector<float> cell_pressure(static_cast<std::size_t>(num_cells), config.p_inf);
  result.last_residual.assign(static_cast<std::size_t>(num_cells) * 5, 0.0f);
  result.last_spectral_radius.assign(static_cast<std::size_t>(num_cells), 0.0f);
  result.residual_magnitude.assign(static_cast<std::size_t>(num_cells), 0.0f);
  const std::vector<int> face_boundary_type = build_face_boundary_type(result.mesh);

  const FreestreamReference reference = {
    rho_inf,
    config.p_inf,
    config.aoa_deg,
    v_inf,
    config.mesh.chord,
    config.x_ref,
    config.y_ref,
  };

  const std::filesystem::path output_dir = config.output_dir.empty() ? "." : config.output_dir;
  std::filesystem::create_directories(output_dir);
  result.residuals_csv_path = output_dir / "residuals.csv";
  result.forces_csv_path = output_dir / "forces.csv";
  result.cp_csv_path = output_dir / "cp_wall.csv";
  result.vtu_path = output_dir / "field_0000.vtu";

  std::ofstream residual_csv(result.residuals_csv_path, std::ios::trunc);
  std::ofstream forces_csv(result.forces_csv_path, std::ios::trunc);
  if (!residual_csv || !forces_csv) {
    throw std::runtime_error("Failed to open Euler CSV outputs.");
  }
  residual_csv << "iter,residual_l1,residual_l2,residual_linf,residual_ratio\n";
  forces_csv << "iter,cl,cd,cm,lift,drag,moment\n";

  float initial_residual_l2 = -1.0f;
  float previous_residual_l2 = -1.0f;
  ForceCoefficients previous_forces;
  int stable_force_iters = 0;
  constexpr int kForceStableWindow = 6;
  float cfl_scale = 1.0f;

#if CFD_HAS_CUDA
  EulerCudaBuffersGuard cuda_buffers_guard;
  if (resolved_backend == "cuda") {
    std::string error_message;
    if (!cfd::cuda_backend::init_euler_device_buffers(
          result.mesh, face_boundary_type, &cuda_buffers_guard.buffers, &error_message)) {
      throw std::runtime_error("CUDA Euler buffer initialization failed: " + error_message);
    }
  }
#endif

  for (int iter = 0; iter < config.iterations; ++iter) {
    for (int cell = 0; cell < num_cells; ++cell) {
      primitive[static_cast<std::size_t>(cell)] =
        conservative_to_primitive(load_conservative_state(result.conserved, cell), gamma);
    }

    const bool use_second_order = (iter >= 25);
    PrimitiveGradients gradients;
    if (use_second_order) {
      gradients = compute_green_gauss_gradients(result.mesh, primitive);
    } else {
      gradients.values.assign(static_cast<std::size_t>(num_cells) * 5 * 2, 0.0f);
    }

    std::fill(residual.begin(), residual.end(), 0.0f);
    std::fill(spectral_radius.begin(), spectral_radius.end(), 0.0f);
    std::fill(result.residual_magnitude.begin(), result.residual_magnitude.end(), 0.0f);
    if (resolved_backend == "cpu") {
      assemble_euler_residual_cpu(result.mesh, primitive, gradients, face_boundary_type,
                                  conservative_inf, gamma, use_second_order, &residual,
                                  &spectral_radius);
    } else {
#if CFD_HAS_CUDA
      cfd::cuda_backend::EulerResidualConfig cuda_config;
      cuda_config.gamma = gamma;
      cuda_config.use_second_order = use_second_order;
      cuda_config.farfield_state = conservative_inf;

      std::string error_message;
      if (!cfd::cuda_backend::euler_residual_cuda(
            result.mesh, cuda_config, &cuda_buffers_guard.buffers, result.conserved.data(),
            gradients.values.data(), residual.data(), spectral_radius.data(), &error_message)) {
        throw std::runtime_error("CUDA Euler residual failed: " + error_message);
      }
#else
      throw std::runtime_error("CUDA backend requested but unavailable.");
#endif
    }
    result.last_residual = residual;
    result.last_spectral_radius = spectral_radius;

    double residual_l1 = 0.0;
    double residual_l2_sum = 0.0;
    double residual_linf = 0.0;
    for (int cell = 0; cell < num_cells; ++cell) {
      const float inv_vol = 1.0f / std::max(result.mesh.cell_volume[cell], 1.0e-12f);
      double cell_mag2 = 0.0;
      for (int k = 0; k < 5; ++k) {
        const float value = residual[static_cast<std::size_t>(5 * cell + k)] * inv_vol;
        const double mag = std::abs(static_cast<double>(value));
        residual_l1 += mag;
        residual_l2_sum += mag * mag;
        residual_linf = std::max(residual_linf, mag);
        cell_mag2 += static_cast<double>(value) * static_cast<double>(value);
      }
      result.residual_magnitude[static_cast<std::size_t>(cell)] = static_cast<float>(std::sqrt(cell_mag2));
    }

    const float residual_l2 =
      static_cast<float>(std::sqrt(residual_l2_sum / std::max(1, num_cells * 5)));
    if (initial_residual_l2 < 0.0f) {
      initial_residual_l2 = std::max(residual_l2, 1.0e-20f);
    }
    const float residual_ratio = residual_l2 / std::max(initial_residual_l2, 1.0e-20f);

    const float cfl = compute_cfl(config, iter) * cfl_scale;
    for (int cell = 0; cell < num_cells; ++cell) {
      const float denom = std::max(spectral_radius[static_cast<std::size_t>(cell)], 1.0e-8f);
      ConservativeState state = load_conservative_state(result.conserved, cell);
      for (int k = 0; k < 5; ++k) {
        const float delta = cfl * residual[static_cast<std::size_t>(5 * cell + k)] / denom;
        if (k == 0) {
          state.rho -= delta;
        } else if (k == 1) {
          state.rhou -= delta;
        } else if (k == 2) {
          state.rhov -= delta;
        } else if (k == 3) {
          state.rhow -= delta;
        } else {
          state.rhoE -= delta;
        }
      }
      enforce_physical_state(&state, gamma);
      store_conservative_state(&result.conserved, cell, state);
      cell_pressure[static_cast<std::size_t>(cell)] = pressure_from_conservative(state, gamma);
    }
    if (previous_residual_l2 > 0.0f) {
      if (residual_l2 > previous_residual_l2 * 1.05f) {
        cfl_scale = std::max(cfl_scale * 0.65f, 0.05f);
      } else if (residual_l2 < previous_residual_l2 * 0.95f) {
        cfl_scale = std::min(cfl_scale * 1.02f, 1.0f);
      }
    }
    previous_residual_l2 = residual_l2;

    result.forces = integrate_pressure_forces(result.mesh, cell_pressure, reference, "wall");
    const float dcl = std::abs(result.forces.cl - previous_forces.cl);
    const float dcd = std::abs(result.forces.cd - previous_forces.cd);
    const float dcm = std::abs(result.forces.cm - previous_forces.cm);
    if (iter > 0 && dcl < config.force_stability_tol && dcd < config.force_stability_tol &&
        dcm < config.force_stability_tol) {
      ++stable_force_iters;
    } else {
      stable_force_iters = 0;
    }
    previous_forces = result.forces;

    EulerIterationRecord record;
    record.iter = iter;
    record.residual_l1 = static_cast<float>(residual_l1);
    record.residual_l2 = residual_l2;
    record.residual_linf = static_cast<float>(residual_linf);
    record.cl = result.forces.cl;
    record.cd = result.forces.cd;
    record.cm = result.forces.cm;
    result.history.push_back(record);

    residual_csv << iter << "," << residual_l1 << "," << residual_l2 << "," << residual_linf << ","
                 << residual_ratio << "\n";
    forces_csv << iter << "," << result.forces.cl << "," << result.forces.cd << ","
               << result.forces.cm << "," << result.forces.lift << "," << result.forces.drag << ","
               << result.forces.moment << "\n";

    const bool residual_ok = residual_ratio <= config.residual_reduction_target;
    const bool force_ok = stable_force_iters >= kForceStableWindow;
    if (iter + 1 >= config.min_iterations && residual_ok && force_ok) {
      break;
    }
  }

  result.rho.assign(static_cast<std::size_t>(num_cells), 0.0f);
  result.u.assign(static_cast<std::size_t>(num_cells), 0.0f);
  result.v.assign(static_cast<std::size_t>(num_cells), 0.0f);
  result.p.assign(static_cast<std::size_t>(num_cells), 0.0f);
  result.mach.assign(static_cast<std::size_t>(num_cells), 0.0f);

  for (int cell = 0; cell < num_cells; ++cell) {
    const ConservativeState state = load_conservative_state(result.conserved, cell);
    const PrimitiveState prim = conservative_to_primitive(state, gamma);
    result.rho[static_cast<std::size_t>(cell)] = prim.rho;
    result.u[static_cast<std::size_t>(cell)] = prim.u;
    result.v[static_cast<std::size_t>(cell)] = prim.v;
    result.p[static_cast<std::size_t>(cell)] = prim.p;
    const float a = speed_of_sound(prim, gamma);
    const float vmag = std::sqrt(prim.u * prim.u + prim.v * prim.v);
    result.mach[static_cast<std::size_t>(cell)] = vmag / std::max(a, 1.0e-8f);
    cell_pressure[static_cast<std::size_t>(cell)] = prim.p;
  }

  result.wall_cp = extract_wall_cp(result.mesh, cell_pressure, reference, "wall");
  {
    std::ofstream cp_csv(result.cp_csv_path, std::ios::trunc);
    if (!cp_csv) {
      throw std::runtime_error("Failed to write cp_wall.csv.");
    }
    cp_csv << "s,x,y,Cp\n";
    for (const auto& sample : result.wall_cp) {
      cp_csv << sample.s << "," << sample.x << "," << sample.y << "," << sample.cp << "\n";
    }
  }

  const bool vtu_ok =
    write_euler_cell_vtu(result.vtu_path, result.mesh, result.rho, result.u, result.v, result.p,
                         result.mach, result.residual_magnitude);
  if (!vtu_ok) {
    throw std::runtime_error("Failed to write Euler VTU output.");
  }

  // TODO(numerics): Add HLLC/Roe options behind the same face-loop interface.
  // TODO(cuda): Keep state and residual resident on GPU across pseudo-time iterations.
  // TODO(cuda): Move gradient computation to GPU.
  // TODO(cuda): Add implicit/JFNK paths that reuse the GPU residual operator.
  return result;
}

EulerRunResult run_euler_airfoil_case_cpu(const EulerAirfoilCaseConfig& config) {
  return run_euler_airfoil_case(config, "cpu");
}
}  // namespace cfd::core
