#include "cfd_core/backend.hpp"
#include "cfd_core/solvers/euler_solver.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>

int main() {
  if (!cfd::core::cuda_available()) {
    std::cout << "CUDA unavailable; skipping Euler CPU/CUDA parity assertion.\n";
    return 0;
  }

  cfd::core::EulerAirfoilCaseConfig config;
  config.mesh.naca_code = "0012";
  config.mesh.num_circumferential = 48;
  config.mesh.num_radial = 10;
  config.mesh.farfield_radius = 10.0f;
  config.mesh.radial_stretch = 1.25f;
  config.gamma = 1.4f;
  config.gas_constant = 1.0f;
  config.p_inf = 0.05f;
  config.t_inf = 1.0f;
  config.rho_inf = 1.0f;
  config.iterations = 26;
  config.min_iterations = 26;
  config.cfl_start = 0.0f;
  config.cfl_max = 0.0f;
  config.cfl_ramp_iters = 1;
  config.residual_reduction_target = 0.0f;
  config.force_stability_tol = 0.0f;

  config.output_dir = std::filesystem::path("euler_parity_cpu");
  const cfd::core::EulerRunResult cpu_result = cfd::core::run_euler_airfoil_case(config, "cpu");

  config.output_dir = std::filesystem::path("euler_parity_cuda");
  const cfd::core::EulerRunResult cuda_result = cfd::core::run_euler_airfoil_case(config, "cuda");

  if (cpu_result.last_residual.size() != cuda_result.last_residual.size()) {
    std::cerr << "Residual vector size mismatch.\n";
    return 2;
  }

  float max_abs_diff = 0.0f;
  for (std::size_t i = 0; i < cpu_result.last_residual.size(); ++i) {
    max_abs_diff = std::max(
      max_abs_diff, std::abs(cpu_result.last_residual[i] - cuda_result.last_residual[i]));
  }

  if (max_abs_diff > 1.0e-5f) {
    std::cerr << "Euler residual CPU/CUDA mismatch too large: " << max_abs_diff << "\n";
    return 3;
  }

  return 0;
}
