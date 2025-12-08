#include "utils.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("evsim", &evsim, "Event simulation (CUDA)",
          py::arg("new_image"), 
          py::arg("new_time"), 
          py::arg("intensity_state_ub"), 
          py::arg("intensity_state_lb"),
          py::arg("event_x_buf"),
          py::arg("event_y_buf"),
          py::arg("event_t_buf"),
          py::arg("event_p_buf"),
          py::arg("MINIMUM_CONTRAST_THRESHOLD_NEG"),
          py::arg("MINIMUM_CONTRAST_THRESHOLD_POS")
    );
}