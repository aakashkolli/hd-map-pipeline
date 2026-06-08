#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace py = pybind11;

namespace {

constexpr std::size_t kPointColumns = 4;
constexpr std::size_t kSpatialColumns = 3;

struct VoxelKey {
  long long x;
  long long y;
  long long z;

  bool operator==(const VoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelKeyHash {
  std::size_t operator()(const VoxelKey& key) const {
    const auto hx = std::hash<long long>{}(key.x);
    const auto hy = std::hash<long long>{}(key.y);
    const auto hz = std::hash<long long>{}(key.z);
    return hx ^ (hy << 1) ^ (hz << 2);
  }
};

}  // namespace

py::array_t<float> voxel_downsample_cpp(
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    double voxel_size) {
  if (voxel_size <= 0.0) {
    throw std::invalid_argument("voxel_size must be positive.");
  }

  const auto input = points.request();
  if (input.ndim != 2 || input.shape[1] != kPointColumns) {
    throw std::invalid_argument("Expected point array shape (N, 4).");
  }

  const auto row_count = static_cast<std::size_t>(input.shape[0]);
  if (row_count == 0) {
    return py::array_t<float>(
        std::vector<py::ssize_t>{0, static_cast<py::ssize_t>(kPointColumns)});
  }

  const auto* data = static_cast<const float*>(input.ptr);
  std::unordered_set<VoxelKey, VoxelKeyHash> seen;
  seen.reserve(row_count);
  std::vector<float> output;
  output.reserve(row_count * kPointColumns);

  for (std::size_t row = 0; row < row_count; ++row) {
    const auto offset = row * kPointColumns;
    const VoxelKey key{
        static_cast<long long>(std::floor(data[offset] / voxel_size)),
        static_cast<long long>(std::floor(data[offset + 1] / voxel_size)),
        static_cast<long long>(std::floor(data[offset + 2] / voxel_size)),
    };

    if (seen.insert(key).second) {
      for (std::size_t column = 0; column < kPointColumns; ++column) {
        output.push_back(data[offset + column]);
      }
    }
  }

  const auto output_rows = output.size() / kPointColumns;
  py::array_t<float> result(std::vector<py::ssize_t>{
      static_cast<py::ssize_t>(output_rows),
      static_cast<py::ssize_t>(kPointColumns)});
  auto result_buffer = result.request();
  auto* result_data = static_cast<float*>(result_buffer.ptr);
  std::copy(output.begin(), output.end(), result_data);
  return result;
}
