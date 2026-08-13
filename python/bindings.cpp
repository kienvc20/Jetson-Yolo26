#include "yolo26seg/trt_engine.hpp"
#include "yolo26seg/yolo26_seg.hpp"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>

namespace py = pybind11;

PYBIND11_MODULE(yolo26seg_runtime, module) {
    module.doc() = "TensorRT 8.2 YOLO26-seg runtime for the original Jetson Nano";

    py::class_<y26::BindingInfo>(module, "BindingInfo")
        .def_readonly("index", &y26::BindingInfo::index)
        .def_readonly("name", &y26::BindingInfo::name)
        .def_readonly("is_input", &y26::BindingInfo::isInput)
        .def_property_readonly("dtype", [](const y26::BindingInfo& binding) {
            return y26::dataTypeName(binding.dataType);
        })
        .def_property_readonly("shape", [](const y26::BindingInfo& binding) {
            std::vector<int> shape;
            shape.reserve(static_cast<std::size_t>(binding.dimensions.nbDims));
            for (int index = 0; index < binding.dimensions.nbDims; ++index) {
                shape.push_back(binding.dimensions.d[index]);
            }
            return shape;
        })
        .def_readonly("bytes", &y26::BindingInfo::bytes);

    py::class_<y26::InferenceStats>(module, "InferenceStats")
        .def_readonly(
            "preprocess_ms",
            &y26::InferenceStats::preprocessMilliseconds)
        .def_readonly(
            "inference_ms",
            &y26::InferenceStats::inferenceMilliseconds);

    py::class_<y26::Yolo26Seg>(module, "Yolo26Seg")
        .def(
            py::init<const std::string&, int, int, const std::string&>(),
            py::arg("engine_path"),
            py::arg("input_width"),
            py::arg("input_height"),
            py::arg("input_name") = "images")
        .def_property_readonly("bindings", &y26::Yolo26Seg::bindings)
        .def("infer", [](
            y26::Yolo26Seg& model,
            py::array_t<
                std::uint8_t,
                py::array::c_style | py::array::forcecast> image) {
            const py::buffer_info info = image.request();
            if (info.ndim != 3 || info.shape[2] != 3) {
                throw std::invalid_argument(
                    "Expected a contiguous HWC uint8 BGR image");
            }

            y26::InferenceStats stats;
            {
                py::gil_scoped_release release;
                stats = model.infer(
                    static_cast<const std::uint8_t*>(info.ptr),
                    static_cast<int>(info.shape[1]),
                    static_cast<int>(info.shape[0]),
                    static_cast<int>(info.strides[0]));
            }
            return stats;
        });
}
