#pragma once

#include "cfd_core/mesh.hpp"
#include "cfd_core/mesh/airfoil_mesh.hpp"
#include "cfd_core/numerics/euler_flux.hpp"
#include "cfd_core/post/forces.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace cfd::core {
struct EulerAirfoilCaseConfig {
  AirfoilMeshConfig mesh;
  int iterations = 400;
  int min_iterations = 40;
  int cfl_ramp_iters = 120;
  float gamma = 1.4f;
  float gas_constant = 287.05f;
  float mach = 0.15f;
  float aoa_deg = 2.0f;
  float p_inf = 101325.0f;
  float t_inf = 288.15f;
  float rho_inf = 0.0f;
  float cfl_start = 0.2f;
  float cfl_max = 1.2f;
  float residual_reduction_target = 1.0e-3f;
  float force_stability_tol = 2.0e-5f;
  float x_ref = 0.25f;
  float y_ref = 0.0f;
  std::filesystem::path output_dir = ".";
};

struct EulerIterationRecord {
  int iter = 0;
  float residual_l1 = 0.0f;
  float residual_l2 = 0.0f;
  float residual_linf = 0.0f;
  float cl = 0.0f;
  float cd = 0.0f;
  float cm = 0.0f;
};

struct EulerRunResult {
  UnstructuredMesh mesh;
  std::vector<float> conserved;  // [cell][rho,rhou,rhov,rhow,rhoE]
  std::vector<float> last_residual;       // [cell][rho,rhou,rhov,rhow,rhoE]
  std::vector<float> last_spectral_radius;
  std::vector<float> residual_magnitude;
  std::vector<float> rho;
  std::vector<float> u;
  std::vector<float> v;
  std::vector<float> p;
  std::vector<float> mach;
  std::vector<EulerIterationRecord> history;
  std::vector<WallCpSample> wall_cp;
  ForceCoefficients forces;
  std::filesystem::path residuals_csv_path;
  std::filesystem::path forces_csv_path;
  std::filesystem::path cp_csv_path;
  std::filesystem::path vtu_path;
};

EulerRunResult run_euler_airfoil_case(const EulerAirfoilCaseConfig& config,
                                      const std::string& backend = "cpu");
EulerRunResult run_euler_airfoil_case_cpu(const EulerAirfoilCaseConfig& config);
}  // namespace cfd::core
