import h5py
import torch
import numpy as np


MAP_BORDER_INDICATOR = 2
RECOLOR_MAP = np.array(
    [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
)  # free, occupied, unknown


def outline_border(top_down_map):
    left_right_block_nav = (top_down_map[:, :-1] == 1) & (
        top_down_map[:, :-1] != top_down_map[:, 1:]
    )
    left_right_nav_block = (top_down_map[:, 1:] == 1) & (
        top_down_map[:, :-1] != top_down_map[:, 1:]
    )

    up_down_block_nav = (top_down_map[:-1] == 1) & (
        top_down_map[:-1] != top_down_map[1:]
    )
    up_down_nav_block = (top_down_map[1:] == 1) & (
        top_down_map[:-1] != top_down_map[1:]
    )

    top_down_map[:, :-1][left_right_block_nav] = MAP_BORDER_INDICATOR
    top_down_map[:, 1:][left_right_nav_block] = MAP_BORDER_INDICATOR

    top_down_map[:-1][up_down_block_nav] = MAP_BORDER_INDICATOR
    top_down_map[1:][up_down_nav_block] = MAP_BORDER_INDICATOR


@torch.compile(mode="reduce-overhead")
def color2intensity(color_im: torch.Tensor) -> torch.Tensor:
    """Convert a color image to an intensity image.

    Args:
        color_im (torch.Tensor): Input color image of shape (H, W, 3) with RGB channels.

    Returns:
        torch.Tensor: Intensity image of shape (H, W).
    """
    intensity_im = (
        0.2989 * color_im[:, :, 0]
        + 0.5870 * color_im[:, :, 1]
        + 0.1140 * color_im[:, :, 2]
    )
    return intensity_im


def get_pose_on_navmesh(scene_bounds, position_3d, meters_per_pixel=0.1) -> np.ndarray:
    """
    Project the current 3D pose onto the 2D navmesh coordinates.

    Args:
        scene_bounds (tuple): ((min_x, min_y, min_z), (max_x, max_y, max_z)) bounds of the scene.
        position_3d (np.ndarray): Current 3D position [x, y, z].
        meters_per_pixel (float): Scale factor to convert meters to pixels on the navmesh.

    Returns:
        np.ndarray: 2D coordinates [px, py] on the navmesh corresponding to the current 3D position.
    """
    # Convert 3D x,z to topdown x,y (y-axis is vertical in Habitat)
    px = (position_3d[0] - scene_bounds[0][0]) / meters_per_pixel
    py = (-position_3d[1] - scene_bounds[0][2]) / meters_per_pixel
    return np.array([px, py], dtype=np.int32)


def init_h5(save_h5, height, width):
    """
    Initialize HDF5 file for storing simulation data.
    """
    h5f = h5py.File(save_h5, "w")
    h5f.create_dataset(
        "events/x",
        (0,),
        maxshape=(None,),
        dtype=np.uint16,
        chunks=(40000,),
        compression="lzf",
    )  # 0-640
    h5f.create_dataset(
        "events/y",
        (0,),
        maxshape=(None,),
        dtype=np.uint16,
        chunks=(40000,),
        compression="lzf",
    )  # 0-480
    h5f.create_dataset(
        "events/p",
        (0,),
        maxshape=(None,),
        dtype=np.uint8,
        chunks=(40000,),
        compression="lzf",
    )  # 0-1
    h5f.create_dataset(
        "events/t",
        (0,),
        maxshape=(None,),
        dtype=np.uint64,
        chunks=(40000,),
        compression="lzf",
    )  # us
    h5f.create_dataset(
        "events/ms_idx",
        (0,),
        maxshape=(None,),
        dtype=np.uint64,
        chunks=(1000,),
        compression="lzf",
    )  # milliseconds to index of events
    h5f.create_dataset(
        "color/data",
        (0, height, width, 3),
        maxshape=(None, height, width, 3),
        dtype=np.uint8,
        chunks=(1, height, width, 3),
        compression="lzf",
    )  # color image
    h5f.create_dataset(
        "color/t",
        (0,),
        maxshape=(None,),
        dtype=np.uint64,
        chunks=(1000,),
        compression="lzf",
    )  # timestamps of color images in microseconds
    h5f.create_dataset(
        "depth/data",
        (0, height, width),
        maxshape=(None, height, width),
        dtype=np.float32,
        chunks=(1, height, width),
        compression="lzf",
    )  # depth image
    h5f.create_dataset(
        "depth/t",
        (0,),
        maxshape=(None,),
        dtype=np.uint64,
        chunks=(1000,),
        compression="lzf",
    )  # timestamps of depth images in microseconds
    h5f.create_dataset(
        "state/x",
        (0, 3),
        maxshape=(None, 3),
        dtype=np.float32,
        chunks=(1000, 3),
        compression="lzf",
    )  # position (x, y, z) #! We store the rotorpy states
    h5f.create_dataset(
        "state/q",
        (0, 4),
        maxshape=(None, 4),
        dtype=np.float32,
        chunks=(1000, 4),
        compression="lzf",
    )  # quaternion (w, x, y, z)
    h5f.create_dataset(
        "state/t",
        (0,),
        maxshape=(None,),
        dtype=np.uint64,
        chunks=(1000,),
        compression="lzf",
    )  # timestamps of state in microseconds
    h5f.create_dataset(
        "imu/accel",
        (0, 3),
        maxshape=(None, 3),
        dtype=np.float32,
        chunks=(1000, 3),
        compression="lzf",
    )  # Accelerometer data (x, y, z)
    h5f.create_dataset(
        "imu/gyro",
        (0, 3),
        maxshape=(None, 3),
        dtype=np.float32,
        chunks=(1000, 3),
        compression="lzf",
    )  # Gyroscope data (x, y, z)
    h5f.create_dataset(
        "imu/t",
        (0,),
        maxshape=(None,),
        dtype=np.uint64,
        chunks=(1000,),
        compression="lzf",
    )  # Timestamps of IMU data in microseconds

    return h5f


def append_data_to_h5(
    h5f, events=None, color=None, depth=None, state=None, imu=None, time=None
):
    """
    Append events, color, and/or depth data to HDF5 file.

    Args:
        h5f: HDF5 file object.
        events: Optional tuple of (x, y, t, p) where:
            - x: Array of x coordinates of events.
            - y: Array of y coordinates of events.
            - t: Array of timestamps of events in microseconds.
            - p: Array of polarities of events (0 or 1).
        color: Optional color image to append.
        depth: Optional depth image to append.
        state: Optional state data to append
            Dictionary with keys:
            - x: Array of position (x, y, z).
            - q: Array of quaternion (w, x, y, z).
        imu: Optional IMU data to append
            Dictionary with keys:
            - accel: Array of accelerometer data (x, y, z).
            - gyro: Array of gyroscope data (x, y, z).
        time: Time in seconds (required).

    Returns:
        int: Number of events appended (0 if no events were provided).
    """
    assert time is not None, "Time must be provided to append data."
    time_us = int(time * 1e6)  # Convert time to microseconds

    if events is not None:
        dset_x = h5f["events/x"]
        dset_y = h5f["events/y"]
        dset_p = h5f["events/p"]
        dset_t = h5f["events/t"]
        dset_ms_idx = h5f["events/ms_idx"]

        cnt = events[0].shape[0]
        new_size = dset_x.shape[0] + cnt
        dset_ms_idx.resize((int(time * 1000) + 1), axis=0)
        dset_ms_idx[-1] = new_size

        if cnt > 0:
            dset_x.resize((new_size), axis=0)
            dset_y.resize((new_size), axis=0)
            dset_t.resize((new_size), axis=0)
            dset_p.resize((new_size), axis=0)
            dset_x[-cnt:] = events[0]
            dset_y[-cnt:] = events[1]
            dset_t[-cnt:] = events[2]
            dset_p[-cnt:] = events[3]

    if state is not None:
        dset_x_state = h5f["state/x"]
        dset_q_state = h5f["state/q"]
        dset_t_state = h5f["state/t"]
        new_size = dset_x_state.shape[0] + 1
        dset_x_state.resize((new_size), axis=0)
        dset_q_state.resize((new_size), axis=0)
        dset_t_state.resize((new_size), axis=0)
        dset_x_state[-1] = state["x"]
        dset_q_state[-1] = state["q"]
        dset_t_state[-1] = time_us  # store time in microseconds

    # Append color image if provided
    if color is not None:
        dset_color = h5f["color/data"]
        dset_color_t = h5f["color/t"]
        new_size = dset_color.shape[0] + 1
        dset_color.resize((new_size), axis=0)
        dset_color_t.resize((new_size), axis=0)
        dset_color[-1] = color
        dset_color_t[-1] = time_us  # convert to microseconds

    # Append depth image if provided
    if depth is not None:
        dset_depth = h5f["depth/data"]
        dset_depth_t = h5f["depth/t"]
        new_size = dset_depth.shape[0] + 1
        dset_depth.resize((new_size), axis=0)
        dset_depth_t.resize((new_size), axis=0)
        dset_depth[-1] = depth
        dset_depth_t[-1] = time_us  # convert to microseconds

    # Append IMU data if provided
    if imu is not None:
        dset_accel = h5f["imu/accel"]
        dset_gyro = h5f["imu/gyro"]
        dset_imu_t = h5f["imu/t"]
        new_size = dset_accel.shape[0] + 1
        dset_accel.resize((new_size), axis=0)
        dset_gyro.resize((new_size), axis=0)
        dset_imu_t.resize((new_size), axis=0)
        dset_accel[-1] = imu["accel"]
        dset_gyro[-1] = imu["gyro"]
        dset_imu_t[-1] = time_us
