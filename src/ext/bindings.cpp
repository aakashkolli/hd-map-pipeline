#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

py::array_t<float> voxel_downsample_cpp(
    py::array_t<float, py::array::c_style | py::array::forcecast> points,
    double voxel_size);

PYBIND11_MODULE(_voxel_filter, module) {
  module.doc() = "C++ voxel grid downsampling extension.";
  module.def(
      "voxel_downsample_cpp",
      &voxel_downsample_cpp,
      py::arg("points"),
      py::arg("voxel_size"));
}
