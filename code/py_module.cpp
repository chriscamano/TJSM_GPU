#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "main.hpp"

namespace py = pybind11;

// Helper function to expose the output buffer as a NumPy array.
py::array_t<uint32_t> get_output_numpy() {
    return py::array_t<uint32_t>(
        { height, width },                            // shape: 2D array (height x width)
        { width * sizeof(uint32_t), sizeof(uint32_t) }, // strides
        output_buffer                                // pointer to the data
    );
}

PYBIND11_MODULE(spectral_density, m) {
    m.doc() = "Python wrapper for the computational functions from main.cpp";
    
    m.def("set_resolution", &set_resolution, "Set the resolution of the output buffer", py::arg("resolution"));
    m.def("set_zoom", &set_zoom, "Set the zoom factor", py::arg("zoom"));
    m.def("set_matrix", &set_matrix, "Set the matrix and compute output", py::arg("dim"), py::arg("pointer"));
    m.def("get_output_buffer", &get_output_buffer, "Get the raw output buffer pointer");
    m.def("get_output_numpy", &get_output_numpy, "Get the output buffer as a NumPy array");
}
