#include "cfd_core/backend.hpp"
#include "cfd_core/io_vtk.hpp"
#include "cfd_core/mesh.hpp"
#include "cfd_core/solvers/euler_solver.hpp"
#include "cfd_core/solver.hpp"
#include "cfd_core/version.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if CFD_HAS_CUDA
#include "cfd_core/cuda_backend.hpp"
#endif

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace {
std::vector<float> make_default_phi(const cfd::core::UnstructuredMesh& mesh) {
  std::vector<float> phi(static_cast<std::size_t>(mesh.num_cells), 1.0f);
  for (int c = 0; c < mesh.num_cells; ++c) {
    const float x = mesh.cell_center[3 * c + 0];
    const float y = mesh.cell_center[3 * c + 1];
    phi[c] = 1.0f + 0.25f * x - 0.15f * y;
  }
  return phi;
}

std::vector<float> compute_scalar_residual_cpu(const cfd::core::UnstructuredMesh& mesh,
                                               const std::vector<float>& phi,
                                               const std::array<float, 3>& u_inf,
                                               const float inflow_phi) {
  const int num_cells = mesh.num_cells;
  const int num_faces = mesh.num_faces;
  int thread_count = 1;

#if defined(_OPENMP)
  thread_count = std::max(1, omp_get_max_threads());
#endif

  std::vector<float> local_accum(static_cast<std::size_t>(thread_count) * num_cells, 0.0f);

#if defined(_OPENMP)
#pragma omp parallel
#endif
  {
    int thread_id = 0;
#if defined(_OPENMP)
    thread_id = omp_get_thread_num();
#endif
    float* residual_local = local_accum.data() + static_cast<std::size_t>(thread_id) * num_cells;

#if defined(_OPENMP)
#pragma omp for schedule(static)
#endif
    for (int face = 0; face < num_faces; ++face) {
      const int owner = mesh.face_owner[face];
      const int neighbor = mesh.face_neighbor[face];
      const float nx = mesh.face_normal[3 * face + 0];
      const float ny = mesh.face_normal[3 * face + 1];
      const float nz = mesh.face_normal[3 * face + 2];
      const float un = (u_inf[0] * nx + u_inf[1] * ny + u_inf[2] * nz) * mesh.face_area[face];

      float upwind_phi = phi[owner];
      if (neighbor >= 0) {
        upwind_phi = (un >= 0.0f) ? phi[owner] : phi[neighbor];
      } else if (un < 0.0f) {
        upwind_phi = inflow_phi;
      }

      const float flux = un * upwind_phi;
      residual_local[owner] -= flux;
      if (neighbor >= 0) {
        residual_local[neighbor] += flux;
      }
    }
  }

  std::vector<float> residual(static_cast<std::size_t>(num_cells), 0.0f);
  for (int thread_id = 0; thread_id < thread_count; ++thread_id) {
    const float* local = local_accum.data() + static_cast<std::size_t>(thread_id) * num_cells;
    for (int c = 0; c < num_cells; ++c) {
      residual[c] += local[c];
    }
  }

  return residual;
}

void normalize_residual_by_volume(const cfd::core::UnstructuredMesh& mesh,
                                  std::vector<float>* residual) {
  if (residual == nullptr) {
    return;
  }
  for (int c = 0; c < mesh.num_cells; ++c) {
    const float volume = mesh.cell_volume[c];
    if (volume > 0.0f) {
      (*residual)[c] /= volume;
    }
  }
}

cfd::core::ScalarResidualNorms compute_norms(const std::vector<float>& residual) {
  cfd::core::ScalarResidualNorms norms;
  double l1 = 0.0;
  double l2_sum = 0.0;
  double linf = 0.0;
  for (const float value : residual) {
    const double mag = std::abs(static_cast<double>(value));
    l1 += mag;
    l2_sum += mag * mag;
    linf = std::max(linf, mag);
  }

  norms.l1 = static_cast<float>(l1);
  norms.l2 = static_cast<float>(std::sqrt(l2_sum));
  norms.linf = static_cast<float>(linf);
  return norms;
}

void write_residuals_csv(const std::filesystem::path& file_path,
                         const cfd::core::ScalarResidualNorms& norms) {
  std::ofstream out(file_path, std::ios::trunc);
  out << "iter,residual,l1,l2,linf\n";
  out << "0," << norms.l2 << "," << norms.l1 << "," << norms.l2 << "," << norms.linf << "\n";
}

void write_forces_csv(const std::filesystem::path& file_path,
                      const cfd::core::ScalarResidualNorms& norms) {
  std::ofstream out(file_path, std::ios::trunc);
  out << "iter,cl,cd,cm\n";
  out << "0," << 0.0f << "," << norms.l2 << "," << 0.0f << "\n";
}

std::string trim_copy(const std::string& value) {
  std::size_t begin = 0;
  while (begin < value.size() && std::isspace(static_cast<unsigned char>(value[begin]))) {
    ++begin;
  }
  std::size_t end = value.size();
  while (end > begin && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
    --end;
  }
  return value.substr(begin, end - begin);
}

std::unordered_map<std::string, std::string> parse_case_kv(const std::filesystem::path& path) {
  std::unordered_map<std::string, std::string> kv;
  std::ifstream in(path);
  if (!in) {
    return kv;
  }

  std::string line;
  while (std::getline(in, line)) {
    const std::string stripped = trim_copy(line);
    if (stripped.empty() || stripped[0] == '#') {
      continue;
    }

    std::size_t sep = stripped.find('=');
    if (sep == std::string::npos) {
      sep = stripped.find(':');
    }
    if (sep == std::string::npos) {
      continue;
    }

    const std::string key = trim_copy(stripped.substr(0, sep));
    const std::string value = trim_copy(stripped.substr(sep + 1));
    if (!key.empty()) {
      kv[key] = value;
    }
  }

  return kv;
}

std::string get_string(const std::unordered_map<std::string, std::string>& kv, const char* key,
                       const std::string& fallback) {
  const auto it = kv.find(key);
  if (it == kv.end() || it->second.empty()) {
    return fallback;
  }
  return it->second;
}

float get_float(const std::unordered_map<std::string, std::string>& kv, const char* key,
                const float fallback) {
  const auto it = kv.find(key);
  if (it == kv.end()) {
    return fallback;
  }
  try {
    return std::stof(it->second);
  } catch (...) {
    return fallback;
  }
}

int get_int(const std::unordered_map<std::string, std::string>& kv, const char* key,
            const int fallback) {
  const auto it = kv.find(key);
  if (it == kv.end()) {
    return fallback;
  }
  try {
    return std::stoi(it->second);
  } catch (...) {
    return fallback;
  }
}

std::array<float, 3> get_vec3(const std::unordered_map<std::string, std::string>& kv,
                              const char* key, const std::array<float, 3>& fallback) {
  const auto it = kv.find(key);
  if (it == kv.end()) {
    return fallback;
  }
  std::array<float, 3> values = fallback;
  std::string text = it->second;
  std::replace(text.begin(), text.end(), ';', ',');
  std::istringstream iss(text);
  std::string token;
  int idx = 0;
  while (std::getline(iss, token, ',') && idx < 3) {
    try {
      values[idx] = std::stof(trim_copy(token));
      ++idx;
    } catch (...) {
      return fallback;
    }
  }
  if (idx < 3) {
    return fallback;
  }
  return values;
}
}  // namespace

namespace cfd::core {
bool cuda_available() {
#if CFD_HAS_CUDA
  return cfd::cuda_backend::cuda_runtime_available(nullptr);
#else
  return false;
#endif
}

std::string normalize_backend(std::string requested_backend) {
  std::transform(requested_backend.begin(), requested_backend.end(), requested_backend.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  if (requested_backend.empty()) {
    requested_backend = "cpu";
  }

  if (requested_backend != "cpu" && requested_backend != "cuda") {
    throw std::invalid_argument("Unsupported backend. Use 'cpu' or 'cuda'.");
  }

  if (requested_backend == "cuda" && !cuda_available()) {
    throw std::runtime_error(
      "CUDA backend requested but unavailable. Build with -DCFD_ENABLE_CUDA=ON and ensure a "
      "CUDA-capable device is visible.");
  }

  return requested_backend;
}

std::string hello() {
  return "cfd_core bindings ok";
}

ScalarRunResult run_scalar_case(const UnstructuredMesh& mesh, const ScalarCaseConfig& config,
                                const std::string& backend) {
  if (mesh.num_cells <= 0 || mesh.num_faces <= 0) {
    throw std::invalid_argument("Mesh must contain cells and faces.");
  }
  if (mesh.face_owner.size() != static_cast<std::size_t>(mesh.num_faces) ||
      mesh.face_neighbor.size() != static_cast<std::size_t>(mesh.num_faces) ||
      mesh.face_area.size() != static_cast<std::size_t>(mesh.num_faces) ||
      mesh.face_normal.size() != static_cast<std::size_t>(mesh.num_faces) * 3 ||
      mesh.face_vertices.size() != static_cast<std::size_t>(mesh.num_faces) * 2 ||
      mesh.face_center.size() != static_cast<std::size_t>(mesh.num_faces) * 3 ||
      mesh.cell_volume.size() != static_cast<std::size_t>(mesh.num_cells)) {
    throw std::invalid_argument("Mesh face/cell array sizes are inconsistent.");
  }

  ScalarRunResult result;
  result.backend = normalize_backend(backend);

  result.phi = config.phi;
  if (result.phi.empty()) {
    result.phi = make_default_phi(mesh);
  }
  if (result.phi.size() != static_cast<std::size_t>(mesh.num_cells)) {
    throw std::invalid_argument("Scalar field phi size must match mesh.num_cells.");
  }

  if (result.backend == "cpu") {
    result.residual =
      compute_scalar_residual_cpu(mesh, result.phi, config.u_inf, config.inflow_phi);
  } else {
#if CFD_HAS_CUDA
    std::string error_message;
    const bool ok = cfd::cuda_backend::compute_scalar_residual_cuda(
      mesh, result.phi, config.u_inf, config.inflow_phi, &result.residual, &error_message);
    if (!ok) {
      throw std::runtime_error("CUDA scalar residual failed: " + error_message);
    }
#else
    throw std::runtime_error("CUDA backend requested but unavailable.");
#endif
  }

  normalize_residual_by_volume(mesh, &result.residual);
  result.residual_norms = compute_norms(result.residual);

  const std::filesystem::path output_dir = config.output_dir.empty() ? "." : config.output_dir;
  std::filesystem::create_directories(output_dir);
  result.residuals_csv_path = output_dir / "residuals.csv";
  result.vtu_path = output_dir / "field_0000.vtu";

  write_residuals_csv(result.residuals_csv_path, result.residual_norms);
  const bool vtu_ok = write_scalar_cell_vtu(result.vtu_path, mesh, result.phi, result.residual);
  if (!vtu_ok) {
    throw std::runtime_error("Failed to write VTU output.");
  }

  return result;
}

RunSummary run_case(const std::string& case_path, const std::string& out_dir,
                    const std::string& backend) {
  const std::string resolved_backend = normalize_backend(backend);
  const std::filesystem::path output_dir(out_dir);
  std::filesystem::create_directories(output_dir);
  const auto case_kv = parse_case_kv(case_path);
  const std::string case_type = get_string(case_kv, "case_type", "scalar_advect_demo");

  const std::filesystem::path run_log_path = output_dir / "run.log";

  if (case_type == "euler_airfoil_2d") {
    EulerAirfoilCaseConfig euler_config;
    euler_config.output_dir = output_dir;
    euler_config.iterations = get_int(case_kv, "iterations", euler_config.iterations);
    euler_config.min_iterations = get_int(case_kv, "min_iterations", euler_config.min_iterations);
    euler_config.cfl_start = get_float(case_kv, "cfl_start", euler_config.cfl_start);
    euler_config.cfl_max = get_float(case_kv, "cfl_max", euler_config.cfl_max);
    euler_config.cfl_ramp_iters = get_int(case_kv, "cfl_ramp_iters", euler_config.cfl_ramp_iters);
    euler_config.residual_reduction_target =
      get_float(case_kv, "residual_reduction_target", euler_config.residual_reduction_target);
    euler_config.force_stability_tol =
      get_float(case_kv, "force_stability_tol", euler_config.force_stability_tol);
    euler_config.gamma = get_float(case_kv, "gamma", euler_config.gamma);
    euler_config.gas_constant = get_float(case_kv, "gas_constant", euler_config.gas_constant);
    euler_config.mach = get_float(case_kv, "mach", euler_config.mach);
    euler_config.aoa_deg = get_float(case_kv, "aoa_deg", euler_config.aoa_deg);
    euler_config.p_inf = get_float(case_kv, "p_inf", euler_config.p_inf);
    euler_config.t_inf = get_float(case_kv, "t_inf", euler_config.t_inf);
    euler_config.rho_inf = get_float(case_kv, "rho_inf", euler_config.rho_inf);
    euler_config.x_ref = get_float(case_kv, "x_ref", euler_config.x_ref);
    euler_config.y_ref = get_float(case_kv, "y_ref", euler_config.y_ref);

    euler_config.mesh.airfoil_source =
      get_string(case_kv, "airfoil_source", euler_config.mesh.airfoil_source);
    euler_config.mesh.naca_code = get_string(case_kv, "naca_code", euler_config.mesh.naca_code);
    euler_config.mesh.coordinate_file =
      get_string(case_kv, "airfoil_file", euler_config.mesh.coordinate_file);
    euler_config.mesh.chord = get_float(case_kv, "chord", euler_config.mesh.chord);
    euler_config.mesh.num_circumferential =
      get_int(case_kv, "num_circumferential", euler_config.mesh.num_circumferential);
    euler_config.mesh.num_radial = get_int(case_kv, "num_radial", euler_config.mesh.num_radial);
    euler_config.mesh.farfield_radius =
      get_float(case_kv, "farfield_radius", euler_config.mesh.farfield_radius);
    euler_config.mesh.radial_stretch =
      get_float(case_kv, "radial_stretch", euler_config.mesh.radial_stretch);

    const EulerRunResult euler_result = run_euler_airfoil_case(euler_config, resolved_backend);
    const EulerIterationRecord& final_record = euler_result.history.back();

    {
      std::ofstream log(run_log_path, std::ios::trunc);
      log << "AeroCFD Euler airfoil 2D\n";
      log << "version=" << version() << "\n";
      log << "case_path=" << case_path << "\n";
      log << "backend=" << resolved_backend << "\n";
      log << "case_type=" << case_type << "\n";
      log << "num_cells=" << euler_result.mesh.num_cells << "\n";
      log << "num_faces=" << euler_result.mesh.num_faces << "\n";
      log << "iterations=" << euler_result.history.size() << "\n";
      log << "residual_l1=" << final_record.residual_l1 << "\n";
      log << "residual_l2=" << final_record.residual_l2 << "\n";
      log << "residual_linf=" << final_record.residual_linf << "\n";
      log << "cl=" << euler_result.forces.cl << "\n";
      log << "cd=" << euler_result.forces.cd << "\n";
      log << "cm=" << euler_result.forces.cm << "\n";
      // TODO(numerics): Add implicit pseudo-time and local Jacobian preconditioning.
      // TODO(physics): Extend from Euler to viscous Navier-Stokes terms.
    }

    RunSummary summary;
    summary.status = "ok";
    summary.backend = resolved_backend;
    summary.case_type = case_type;
    summary.run_log = run_log_path.string();
    summary.iterations = static_cast<int>(euler_result.history.size());
    summary.residual_l1 = final_record.residual_l1;
    summary.residual_l2 = final_record.residual_l2;
    summary.residual_linf = final_record.residual_linf;
    summary.cl = euler_result.forces.cl;
    summary.cd = euler_result.forces.cd;
    summary.cm = euler_result.forces.cm;
    return summary;
  }

  const UnstructuredMesh mesh = make_demo_tri_mesh_2x2();
  ScalarCaseConfig config;
  config.u_inf = get_vec3(case_kv, "u_inf", {1.0f, 0.35f, 0.0f});
  config.inflow_phi = get_float(case_kv, "inflow_phi", 1.0f);
  config.phi = make_default_phi(mesh);
  config.output_dir = output_dir;

  const ScalarRunResult scalar_result = run_scalar_case(mesh, config, resolved_backend);

  const std::filesystem::path forces_path = output_dir / "forces.csv";
  {
    std::ofstream log(run_log_path, std::ios::trunc);
    log << "AeroCFD scalar advection demo\n";
    log << "version=" << version() << "\n";
    log << "case_path=" << case_path << "\n";
    log << "backend=" << resolved_backend << "\n";
    log << "case_type=scalar_advect_demo\n";
    log << "num_cells=" << mesh.num_cells << "\n";
    log << "num_faces=" << mesh.num_faces << "\n";
    log << "residual_l1=" << scalar_result.residual_norms.l1 << "\n";
    log << "residual_l2=" << scalar_result.residual_norms.l2 << "\n";
    log << "residual_linf=" << scalar_result.residual_norms.linf << "\n";
  }

  write_forces_csv(forces_path, scalar_result.residual_norms);

  RunSummary summary;
  summary.status = "ok";
  summary.backend = resolved_backend;
  summary.case_type = "scalar_advect_demo";
  summary.run_log = run_log_path.string();
  summary.iterations = 1;
  summary.residual_l1 = scalar_result.residual_norms.l1;
  summary.residual_l2 = scalar_result.residual_norms.l2;
  summary.residual_linf = scalar_result.residual_norms.linf;
  summary.cl = 0.0f;
  summary.cd = scalar_result.residual_norms.l2;
  summary.cm = 0.0f;
  return summary;
}
}  // namespace cfd::core
