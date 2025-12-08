#include "utils.h"

#define FULL_MASK 0xffffffff

template <typename scalar_t>
__global__ void evsim_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> new_image,
    const uint64_t new_time,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> intensity_state_ub,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> intensity_state_lb,
    torch::PackedTensorAccessor32<uint16_t, 1, torch::RestrictPtrTraits> event_x_buf,
    torch::PackedTensorAccessor32<uint16_t, 1, torch::RestrictPtrTraits> event_y_buf,
    torch::PackedTensorAccessor32<uint64_t, 1, torch::RestrictPtrTraits> event_t_buf,
    torch::PackedTensorAccessor32<uint8_t, 1, torch::RestrictPtrTraits> event_p_buf,
    int32_t* event_count,
    const float MINIMUM_CONTRAST_THRESHOLD_NEG,
    const float MINIMUM_CONTRAST_THRESHOLD_POS,
    const uint32_t MAX_EVENTS,
    const uint16_t height,
    const uint16_t width
){
    const int32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t y = blockIdx.y * blockDim.y + threadIdx.y;

    bool has_event = false;
    bool pos_event = false;

    if (x < width && y < height)
    {
        const scalar_t current_log_intensity = log(new_image[y][x]);
        const scalar_t state_ub = intensity_state_ub[y][x];
        const scalar_t state_lb = intensity_state_lb[y][x];

        pos_event = current_log_intensity > state_ub;
        const bool neg_event = current_log_intensity < state_lb;
        has_event = pos_event || neg_event;

        if (has_event)
        {
            intensity_state_ub[y][x] = current_log_intensity + static_cast<scalar_t>(MINIMUM_CONTRAST_THRESHOLD_POS);
            intensity_state_lb[y][x] = current_log_intensity - static_cast<scalar_t>(MINIMUM_CONTRAST_THRESHOLD_NEG);
        }
        else
        {
            intensity_state_ub[y][x] = min(state_ub, current_log_intensity + static_cast<scalar_t>(MINIMUM_CONTRAST_THRESHOLD_POS));
            intensity_state_lb[y][x] = max(state_lb, current_log_intensity - static_cast<scalar_t>(MINIMUM_CONTRAST_THRESHOLD_NEG));
        }
    }

    // Warp-level aggregation of events to reduce atomic contention
    const uint32_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int8_t lane_id = tid & 31;  // threadIdx.x % 32 max 32
    const int8_t warp_id = tid >> 5;  // threadIdx.x / 32 max ~64

    const uint32_t warp_event_mask = __ballot_sync(FULL_MASK, has_event);
    const int32_t warp_event_count = __popc(warp_event_mask);

    if (warp_event_count > 0)
    {
        int32_t warp_base_idx = 0;
        if (lane_id == 0)
            warp_base_idx = atomicAdd(event_count, warp_event_count);

        // Broadcast the base index to all threads in the warp
        warp_base_idx = __shfl_sync(FULL_MASK, warp_base_idx, 0);

        if (has_event)
        {
            const uint32_t lane_mask = (1u << lane_id) - 1u;
            const uint32_t preceding_events_mask = warp_event_mask & lane_mask;
            const int32_t thread_event_idx = __popc(preceding_events_mask);

            const int32_t global_event_idx = warp_base_idx + thread_event_idx;

            if (global_event_idx < MAX_EVENTS)
            {
                event_x_buf[global_event_idx] = static_cast<uint16_t>(x);
                event_y_buf[global_event_idx] = static_cast<uint16_t>(y);
                event_t_buf[global_event_idx] = new_time;
                event_p_buf[global_event_idx] = pos_event ? 1 : 0;
            }
        }
    }
}

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
    const float MINIMUM_CONTRAST_THRESHOLD_NEG,
    const float MINIMUM_CONTRAST_THRESHOLD_POS
){
    CHECK_CUDA(new_image);
    CHECK_CUDA(intensity_state_ub);
    CHECK_CUDA(intensity_state_lb);
    CHECK_CUDA(event_x_buf);
    CHECK_CUDA(event_y_buf);
    CHECK_CUDA(event_t_buf);
    CHECK_CUDA(event_p_buf);
    
    CHECK_CONTIGUOUS(new_image);
    CHECK_CONTIGUOUS(intensity_state_ub);
    CHECK_CONTIGUOUS(intensity_state_lb);
    CHECK_CONTIGUOUS(event_x_buf);
    CHECK_CONTIGUOUS(event_y_buf);
    CHECK_CONTIGUOUS(event_t_buf);
    CHECK_CONTIGUOUS(event_p_buf);
    
    CHECK_IS_FLOATING(new_image);
    CHECK_IS_FLOATING(intensity_state_ub);
    CHECK_IS_FLOATING(intensity_state_lb);

    TORCH_CHECK(new_image.dim() == 2, "new_image must be 2D (H, W)");
    TORCH_CHECK(intensity_state_ub.dim() == 2, "intensity_state_ub must be 2D (H, W)");
    TORCH_CHECK(intensity_state_lb.dim() == 2, "intensity_state_lb must be 2D (H, W)");

    const uint16_t height = new_image.size(0);
    const uint16_t width = new_image.size(1);
    const uint32_t MAX_EVENTS = event_x_buf.size(0);

    auto event_count = torch::zeros({1}, torch::dtype(torch::kInt32).device(new_image.device()));

    const dim3 threads(32, 32);
    const dim3 blocks(BLOCKS(width, threads.x), BLOCKS(height, threads.y));

    AT_DISPATCH_FLOATING_TYPES(new_image.scalar_type(), "evsim_cuda",
    ([&] {
        evsim_kernel<scalar_t><<<blocks, threads>>>(
            new_image.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            new_time,
            intensity_state_ub.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            intensity_state_lb.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            event_x_buf.packed_accessor32<uint16_t, 1, torch::RestrictPtrTraits>(),
            event_y_buf.packed_accessor32<uint16_t, 1, torch::RestrictPtrTraits>(),
            event_t_buf.packed_accessor32<uint64_t, 1, torch::RestrictPtrTraits>(),
            event_p_buf.packed_accessor32<uint8_t, 1, torch::RestrictPtrTraits>(),
            event_count.data_ptr<int32_t>(),
            MINIMUM_CONTRAST_THRESHOLD_NEG,
            MINIMUM_CONTRAST_THRESHOLD_POS,
            MAX_EVENTS,
            height,
            width
        );
    }));

    auto cuda_error = cudaGetLastError();
    TORCH_CHECK(cuda_error == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(cuda_error));

    cudaDeviceSynchronize();

    const int32_t num_events = std::min(event_count[0].item<int32_t>(), static_cast<int32_t>(MAX_EVENTS));

    if (num_events == 0)
    {
        return std::make_tuple(
            torch::empty({0}, torch::dtype(torch::kUInt16).device(new_image.device())),
            torch::empty({0}, torch::dtype(torch::kUInt16).device(new_image.device())),
            torch::empty({0}, torch::dtype(torch::kUInt64).device(new_image.device())),
            torch::empty({0}, torch::dtype(torch::kUInt8).device(new_image.device()))
        );
    }

    return std::make_tuple(
        event_x_buf.slice(0, 0, num_events),
        event_y_buf.slice(0, 0, num_events),
        event_t_buf.slice(0, 0, num_events),
        event_p_buf.slice(0, 0, num_events)
    );
}