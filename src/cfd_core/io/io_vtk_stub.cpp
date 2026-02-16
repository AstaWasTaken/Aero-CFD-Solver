#include "cfd_core/io_vtk.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <type_traits>

namespace cfd::core {
namespace {
template <typename T>
void write_ascii_value(std::ofstream& out, const T& value) {
  if constexpr (std::is_same_v<T, std::uint8_t> || std::is_same_v<T, std::int8_t> ||
                std::is_same_v<T, char> || std::is_same_v<T, signed char> ||
                std::is_same_v<T, unsigned char>) {
    out << static_cast<int>(value);
  } else {
    out << value;
  }
}

template <typename T>
void write_ascii_data_array(std::ofstream& out, const std::vector<T>& values) {
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out << ' ';
    }
    write_ascii_value(out, values[i]);
  }
}
}  // namespace

bool write_scalar_cell_vtu(const std::filesystem::path& output_path, const UnstructuredMesh& mesh,
                           const std::vector<float>& phi, const std::vector<float>& residual) {
  if (mesh.num_cells <= 0 || mesh.num_faces <= 0) {
    return false;
  }
  if (phi.size() != static_cast<std::size_t>(mesh.num_cells) ||
      residual.size() != static_cast<std::size_t>(mesh.num_cells)) {
    return false;
  }
  if (mesh.points.size() % 3 != 0) {
    return false;
  }

  std::ofstream out(output_path, std::ios::trunc);
  if (!out) {
    return false;
  }

  std::vector<float> residual_mag(residual.size(), 0.0f);
  for (std::size_t i = 0; i < residual.size(); ++i) {
    residual_mag[i] = std::abs(residual[i]);
  }

  const int num_points = static_cast<int>(mesh.points.size() / 3);

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  out << "  <UnstructuredGrid>\n";
  out << "    <Piece NumberOfPoints=\"" << num_points << "\" NumberOfCells=\"" << mesh.num_cells
      << "\">\n";
  out << "      <Points>\n";
  out << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">";
  write_ascii_data_array(out, mesh.points);
  out << "</DataArray>\n";
  out << "      </Points>\n";
  out << "      <Cells>\n";
  out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">";
  write_ascii_data_array(out, mesh.cell_connectivity);
  out << "</DataArray>\n";
  out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">";
  write_ascii_data_array(out, mesh.cell_offsets);
  out << "</DataArray>\n";
  out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">";
  write_ascii_data_array(out, mesh.cell_types);
  out << "</DataArray>\n";
  out << "      </Cells>\n";
  out << "      <PointData/>\n";
  out << "      <CellData Scalars=\"phi\">\n";
  out << "        <DataArray type=\"Float32\" Name=\"phi\" format=\"ascii\">";
  write_ascii_data_array(out, phi);
  out << "</DataArray>\n";
  out << "        <DataArray type=\"Float32\" Name=\"residual_magnitude\" format=\"ascii\">";
  write_ascii_data_array(out, residual_mag);
  out << "</DataArray>\n";
  out << "      </CellData>\n";
  out << "    </Piece>\n";
  out << "  </UnstructuredGrid>\n";
  out << "</VTKFile>\n";

  return true;
}

bool write_euler_cell_vtu(const std::filesystem::path& output_path, const UnstructuredMesh& mesh,
                          const std::vector<float>& rho, const std::vector<float>& u,
                          const std::vector<float>& v, const std::vector<float>& p,
                          const std::vector<float>& mach,
                          const std::vector<float>& residual_magnitude) {
  if (mesh.num_cells <= 0 || mesh.num_faces <= 0) {
    return false;
  }
  const std::size_t n = static_cast<std::size_t>(mesh.num_cells);
  if (rho.size() != n || u.size() != n || v.size() != n || p.size() != n || mach.size() != n ||
      residual_magnitude.size() != n) {
    return false;
  }
  if (mesh.points.size() % 3 != 0) {
    return false;
  }

  std::ofstream out(output_path, std::ios::trunc);
  if (!out) {
    return false;
  }

  const int num_points = static_cast<int>(mesh.points.size() / 3);

  out << "<?xml version=\"1.0\"?>\n";
  out << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  out << "  <UnstructuredGrid>\n";
  out << "    <Piece NumberOfPoints=\"" << num_points << "\" NumberOfCells=\"" << mesh.num_cells
      << "\">\n";
  out << "      <Points>\n";
  out << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">";
  write_ascii_data_array(out, mesh.points);
  out << "</DataArray>\n";
  out << "      </Points>\n";
  out << "      <Cells>\n";
  out << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">";
  write_ascii_data_array(out, mesh.cell_connectivity);
  out << "</DataArray>\n";
  out << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">";
  write_ascii_data_array(out, mesh.cell_offsets);
  out << "</DataArray>\n";
  out << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"ascii\">";
  write_ascii_data_array(out, mesh.cell_types);
  out << "</DataArray>\n";
  out << "      </Cells>\n";
  out << "      <PointData/>\n";
  out << "      <CellData Scalars=\"rho\">\n";
  out << "        <DataArray type=\"Float32\" Name=\"rho\" format=\"ascii\">";
  write_ascii_data_array(out, rho);
  out << "</DataArray>\n";
  out << "        <DataArray type=\"Float32\" Name=\"u\" format=\"ascii\">";
  write_ascii_data_array(out, u);
  out << "</DataArray>\n";
  out << "        <DataArray type=\"Float32\" Name=\"v\" format=\"ascii\">";
  write_ascii_data_array(out, v);
  out << "</DataArray>\n";
  out << "        <DataArray type=\"Float32\" Name=\"p\" format=\"ascii\">";
  write_ascii_data_array(out, p);
  out << "</DataArray>\n";
  out << "        <DataArray type=\"Float32\" Name=\"Mach\" format=\"ascii\">";
  write_ascii_data_array(out, mach);
  out << "</DataArray>\n";
  out << "        <DataArray type=\"Float32\" Name=\"residual_magnitude\" format=\"ascii\">";
  write_ascii_data_array(out, residual_magnitude);
  out << "</DataArray>\n";
  out << "      </CellData>\n";
  out << "    </Piece>\n";
  out << "  </UnstructuredGrid>\n";
  out << "</VTKFile>\n";
  return true;
}
}  // namespace cfd::core
