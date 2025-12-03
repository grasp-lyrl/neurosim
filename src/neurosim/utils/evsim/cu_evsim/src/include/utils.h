#include <torch/extension.h>

// MACROS
#define BLOCKS(N, T) (N + T - 1) / T // rounds up the division

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

// Main dispatch function declaration
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
evsim(
    const torch::Tensor new_image,
    const uint64_t new_time,
    torch::Tensor intensity_state_ub,
    torch::Tensor intensity_state_lb,
    torch::Tensor event_x_buf,
    torch::Tensor event_y_buf,
    torch::Tensor event_t_buf,
    torch::Tensor event_p_buf,
    const float MINIMUM_CONTRAST_THRESHOLD_NEG = 0.35f,
    const float MINIMUM_CONTRAST_THRESHOLD_POS = 0.35f
);
