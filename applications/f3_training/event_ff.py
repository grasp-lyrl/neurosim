"""
EventPatchFF Model Implementation for F3 (Fast Feature Fields)

This is a self-contained implementation of the EventPatchFF model from the F3 paper,
adapted for use with the neurosim OnlineDataLoader.

Reference: https://github.com/grasp-lyrl/fast-feature-fields
"""

import yaml
import logging
from copy import copy

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger("__main__")


class PixelShuffleUpsample(nn.Module):
    """PixelShuffle Upsampling layer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        upscale_factor (int): Upscaling factor. Default: 2.

    Returns:
        torch.Tensor: Upsampled tensor.
    """

    def __init__(self, in_channels, out_channels, upscale_factor=2, kernel_size=3):
        super(PixelShuffleUpsample, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * (upscale_factor**2),
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
        )
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class LayerNorm(nn.Module):
    """LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """GRN (Global Response Normalization) layer"""

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, kernel_size=7, bottleneck=4, dilation=1):
        super().__init__()
        if kernel_size > 1:
            padding = (kernel_size - 1) // 2 * dilation
            self.dwconv = nn.Conv2d(
                dim,
                dim,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=dim,
            )  # depthwise conv
        else:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=1)  # pointwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, bottleneck * dim
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(bottleneck * dim)
        self.pwconv2 = nn.Linear(bottleneck * dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        return input + x


class MultiResolutionHashEncoder(nn.Module):
    """
    I write separate forward functions for 2D and 3D because I can't put if statements inside the forward
    pass, otherwise torch.compile will throw all sorts of errors.
    """

    def __init__(
        self,
        D: int = 3,  # Number of dimensions (2D or 3D)
        L_H: int = 2,  # Number of hashed levels
        L_NH: int = 6,  # Number of non-hashed levels
        levels: int = 8,  # Total number of levels (L_H + L_NH)
        polarity: bool = False,  # Whether to use polarity or not
        feature_size: int = 4,  # Feature size per level
        resolutions: torch.Tensor | None = None,  # L x D tensor of resolutions
        log2_entries_per_level: int = 19,  # log2 of number of entries per level
    ):
        super().__init__()

        self.PI1: int = 1
        self.PI2: int = 2654435761
        self.PI3: int = 805459861

        self.D = D
        self.L_H = L_H
        self.L_NH = L_NH
        self.polarity = polarity
        self.feature_size = feature_size
        self.resolutions = resolutions  # L x D
        self.levels = levels  # Should be equal to L_H + L_NH
        self.log2_entries_per_level = log2_entries_per_level

        try:
            self.index = getattr(self, f"index{self.D}d")
            self.forward = getattr(
                self, f"forward{'_pol' if self.polarity else '_nopol'}"
            )
        except AttributeError:
            raise ValueError(f"Invalid number of dimensions: {self.D}")
        logger.info(
            f"Using {self.D}D Multi-Resolution Hash Encoder with {self.L_H} hashed levels and {self.L_NH} non-hashed levels and polarity: {self.polarity}"
        )

        def get_hashmap():
            with torch.no_grad():
                hashmap = torch.zeros(
                    (self.levels, 1 << self.log2_entries_per_level, self.feature_size),
                    dtype=torch.float32,
                )
                hashmap.uniform_(-1e-4, 1e-4)
                hashmap = nn.Parameter(
                    hashmap
                )  # L x T x F where T = 2^log2_entries_per_level, F = feature_size
            return hashmap

        # build the hash tables
        if not self.polarity:
            self.hashmap = get_hashmap()
        else:
            self.hashmap_neg = get_hashmap()
            self.hashmap_pos = get_hashmap()

    def hash_linear_congruential3d(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        log2_entries_per_level: int,
    ) -> torch.Tensor:
        return (x * self.PI1 ^ y * self.PI2 ^ t * self.PI3) % (
            1 << log2_entries_per_level
        )

    def hash_linear_congruential2d(
        self, x: torch.Tensor, y: torch.Tensor, log2_entries_per_level: int
    ) -> torch.Tensor:
        return (x * self.PI1 ^ y * self.PI2) % (1 << log2_entries_per_level)

    def index3d(self, eventBlock: torch.Tensor) -> torch.Tensor:
        x, y, t = eventBlock[:, :, 0], eventBlock[:, :, 1], eventBlock[:, :, 2]

        #! Asserts don't work with torch.compile
        if not self.compile:
            assert x.shape == y.shape == t.shape, (
                f"x.shape: {x.shape}, y.shape: {y.shape}, t.shape: {t.shape}"
            )
            assert x.min() >= 0 and x.max() <= 1, "x coordinate should be in [0, 1]"
            assert y.min() >= 0 and y.max() <= 1, "y coordinate should be in [0, 1]"
            assert t.min() >= 0, "t should be non-negative"

        scaled_x = x.unsqueeze(-1) * self.resolutions[:, 0]  # B x N x L
        scaled_y = y.unsqueeze(-1) * self.resolutions[:, 1]  # B x N x L
        scaled_t = t.unsqueeze(-1) * self.resolutions[:, 2]  # B x N x L

        floor_scaled_x = scaled_x.int()  # B x N x L
        floor_scaled_y = scaled_y.int()  # B x N x L
        floor_scaled_t = scaled_t.int()  # B x N x L

        ceil_scaled_x = torch.min(
            floor_scaled_x + 1, self.resolutions[:, 0][None, None, :]
        )  # B x N x L
        ceil_scaled_y = torch.min(
            floor_scaled_y + 1, self.resolutions[:, 1][None, None, :]
        )  # B x N x L
        ceil_scaled_t = torch.min(
            floor_scaled_t + 1, self.resolutions[:, 2][None, None, :]
        )  # B x N x L

        # all combinations of the 8 corners of the cube B X N x L x 8 x 3
        corners = torch.stack(
            [
                torch.stack([floor_scaled_x, floor_scaled_y, floor_scaled_t], dim=-1),
                torch.stack([floor_scaled_x, floor_scaled_y, ceil_scaled_t], dim=-1),
                torch.stack([floor_scaled_x, ceil_scaled_y, floor_scaled_t], dim=-1),
                torch.stack([floor_scaled_x, ceil_scaled_y, ceil_scaled_t], dim=-1),
                torch.stack([ceil_scaled_x, floor_scaled_y, floor_scaled_t], dim=-1),
                torch.stack([ceil_scaled_x, floor_scaled_y, ceil_scaled_t], dim=-1),
                torch.stack([ceil_scaled_x, ceil_scaled_y, floor_scaled_t], dim=-1),
                torch.stack([ceil_scaled_x, ceil_scaled_y, ceil_scaled_t], dim=-1),
            ],
            dim=-2,
        ).int()  # B x N x L x 8 x 3

        # calculate the weights for each corner B x N x L x 8
        weights = torch.prod(
            1
            - (
                corners
                - torch.stack([scaled_x, scaled_y, scaled_t], dim=-1).unsqueeze(-2)
            ).abs(),
            dim=-1,
        )

        # Calculate the indices for the hash table (Hash + Non-hash levels depending on the resolution)
        # B x N x L_H x 8 where L_H is the number of levels that need hashing
        hash_values = self.hash_linear_congruential3d(
            corners[:, :, -self.L_H :, :, 0],
            corners[:, :, -self.L_H :, :, 1],
            corners[:, :, -self.L_H :, :, 2],
            self.log2_entries_per_level,
        )
        # B x N x L_NH x 8 where L_NH is the number of levels that don't need hashing
        nonhash_values = (
            corners[:, :, : self.L_NH, :, 0]
            + corners[:, :, : self.L_NH, :, 1]
            * self.resolutions[: self.L_NH, 0][None, None, :, None]
            + corners[:, :, : self.L_NH, :, 2]
            * (self.resolutions[: self.L_NH, 0] * self.resolutions[: self.L_NH, 1])[
                None, None, :, None
            ]
        )

        return hash_values, nonhash_values, weights

    def index2d(self, eventBlock: torch.Tensor) -> torch.Tensor:
        x, y = eventBlock[:, :, 0], eventBlock[:, :, 1]

        #! Asserts don't work with torch.compile
        if not self.compile:
            assert x.shape == y.shape, f"x.shape: {x.shape}, y.shape: {y.shape}"
            assert x.min() >= 0 and x.max() <= 1, "x coordinate should be in [0, 1]"
            assert y.min() >= 0 and y.max() <= 1, "y coordinate should be in [0, 1]"

        scaled_x = x.unsqueeze(-1) * self.resolutions[:, 0]
        scaled_y = y.unsqueeze(-1) * self.resolutions[:, 1]

        floor_scaled_x = scaled_x.int()
        floor_scaled_y = scaled_y.int()

        ceil_scaled_x = torch.min(
            floor_scaled_x + 1, self.resolutions[:, 0][None, None, :]
        )
        ceil_scaled_y = torch.min(
            floor_scaled_y + 1, self.resolutions[:, 1][None, None, :]
        )

        corners = torch.stack(
            [
                torch.stack([floor_scaled_x, floor_scaled_y], dim=-1),
                torch.stack([floor_scaled_x, ceil_scaled_y], dim=-1),
                torch.stack([ceil_scaled_x, floor_scaled_y], dim=-1),
                torch.stack([ceil_scaled_x, ceil_scaled_y], dim=-1),
            ],
            dim=-2,
        ).int()

        weights = torch.prod(
            1
            - (corners - torch.stack([scaled_x, scaled_y], dim=-1).unsqueeze(-2)).abs(),
            dim=-1,
        )

        hash_values = self.hash_linear_congruential2d(
            corners[:, :, -self.L_H :, :, 0],
            corners[:, :, -self.L_H :, :, 1],
            self.log2_entries_per_level,
        )
        nonhash_values = (
            corners[:, :, : self.L_NH, :, 0]
            + corners[:, :, : self.L_NH, :, 1]
            * self.resolutions[: self.L_NH, 0][None, None, :, None]
        )

        return hash_values, nonhash_values, weights

    def forward_pol(self, eventBlock: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Multi-Resolution Hash Encoder for 4D events (i.e. polarities).


        Args:
            x: x coordinate of the event rescaled to [0, 1]. (B,N)
            y: y coordinate of the event rescaled to [0, 1]. (B,N)
            t: time bin of the event. (B,N)
            (Say if each bucket is 20us, and there's a total of 1000 buckets, then t is in [0, 1000])
            p: polarity of the event. (B,N)

            (x,y,t,p) or (x,y,p)

        Returns:
            The feature field of the events. (B, N, L*F) where L is the number of levels and F is the feature size.
        """
        p = eventBlock[:, :, -1].reshape(-1)
        B, N = eventBlock.shape[0], eventBlock.shape[1]
        hash_values, nonhash_values, weights = self.index(eventBlock)

        hash_values, nonhash_values = (
            hash_values.reshape(B * N, self.L_H, 2**self.D),
            nonhash_values.reshape(B * N, self.L_NH, 2**self.D),
        )
        neg_idx, pos_idx = (
            (p == 0).nonzero().squeeze(),
            (p == 1).nonzero().squeeze(),
        )  #! Boolean indexing doesn't seem to work with torch.compile
        hashmap_features = torch.zeros(
            (B * N, self.levels, 2**self.D, self.feature_size),
            dtype=self.hashmap_neg.dtype,
            device=eventBlock.device,
        )
        for i in range(self.L_NH):
            hashmap_features[neg_idx, i, :, :] = self.hashmap_neg[i][
                nonhash_values[neg_idx, i, :]
            ]
            hashmap_features[pos_idx, i, :, :] = self.hashmap_pos[i][
                nonhash_values[pos_idx, i, :]
            ]
        for i in range(self.L_H):
            hashmap_features[neg_idx, i + self.L_NH, :, :] = self.hashmap_neg[
                i + self.L_NH
            ][hash_values[neg_idx, i, :]]
            hashmap_features[pos_idx, i + self.L_NH, :, :] = self.hashmap_pos[
                i + self.L_NH
            ][hash_values[pos_idx, i, :]]
        hashmap_features = hashmap_features.reshape(
            B, N, self.levels, 2**self.D, self.feature_size
        )

        interpolated_features = torch.sum(
            weights.unsqueeze(-1) * hashmap_features, dim=-2
        )
        interpolated_features = interpolated_features.reshape(B, N, -1)
        return interpolated_features

    def forward_nopol(self, eventBlock: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Multi-Resolution Hash Encoder.

        Args:
            x: x coordinate of the event rescaled to [0, 1]. (B,N)
            y: y coordinate of the event rescaled to [0, 1]. (B,N)
            t: time bin of the event. (B,N)
            (Say if each bucket is 20us, and there's a total of 1000 buckets, then t is in [0, 1000])

            (x,y,t) or (x,y)

        Returns:
            The feature field of the events. (B, N, L*F) where L is the number of levels and F is the feature size.
        """
        B, N = (
            eventBlock.shape[0],
            eventBlock.shape[1],
        )  # Batch size, Number of events in each batch
        hash_values, nonhash_values, weights = self.index(eventBlock)

        # We have a hashmap of size L x T x F where T = 2^log2_entries_per_level, F = feature_size
        # We want to index it with N x L x 8 index tensor. That is, each row in the 1st dim of the index matrix,
        # index into the rows of the hashmap tensor. The 8 corners index into the 1st dim of the hashmap tensor.
        # We don't want to use gather with expand, because in backward pass it goes OOM. Well I don't like loops,
        # but I couldn't find a cleverer way to use gather which doesn't either expand on the N or F dimension.
        hashmap_features = torch.zeros(
            (B, N, self.levels, 2**self.D, self.feature_size),
            dtype=self.hashmap.dtype,
            device=eventBlock.device,
        )
        for i in range(self.L_NH):
            hashmap_features[:, :, i, :, :] = self.hashmap[i][
                nonhash_values[:, :, i, :]
            ]
        for i in range(self.L_H):
            hashmap_features[:, :, i + self.L_NH, :, :] = self.hashmap[i + self.L_NH][
                hash_values[:, :, i, :]
            ]

        interpolated_features = torch.sum(
            weights.unsqueeze(-1) * hashmap_features, dim=-2
        )  # B x N x L x F
        interpolated_features = interpolated_features.reshape(B, N, -1)  # B x N x (L*F)
        return interpolated_features


class EventPatchFF(nn.Module):
    """
    Event Feature Field with Patch Prediction.

    Args:
        multi_hash_encoder: Configuration for the hash encoder
        T: Number of time bins to predict
        frame_sizes: [width, height, time_bins] of input
        dims: Feature dimensions at each stage
        convkernels: Kernel sizes at each stage
        convdepths: Number of blocks at each stage
        convbtlncks: Bottleneck factors at each stage
        convdilations: Dilation factors at each stage
        dskernels: Downsample kernel sizes
        dsstrides: Downsample strides
        patch_size: Spatial patch size for prediction
        use_upsampling: Whether to use upsampling decoder
        upsampling_dims: Output dimensions after upsampling
        use_decoder_block: Whether to use additional decoder block
        return_logits: Whether to return prediction logits
        return_feat: Whether to return intermediate features
        return_loss: Whether to compute and return loss
        loss_fn: Loss function name
        variable_mode: Whether to use variable-length event batches
        device: Device to use
    """

    def __init__(
        self,
        multi_hash_encoder: dict,  # Args for the multi_hash_encoder
        T=20,  # Number of frames to predict
        frame_sizes: list[int] = [1280, 720, 20],  # Input size
        dims: list[int] = [96, 128],  # Feature dimensions at each stage
        convkernels: list[int] = [7, 5],  # Kernel size at each stage
        convdepths: list[int] = [3, 3],  # Number of blocks at each stage
        convbtlncks: list[int] = [4, 4],  # Bottleneck factor
        convdilations: list[int] = [1, 1],  # Dilation factor
        dskernels: list[int] = [5, 5],  # Downsample kernel size
        dsstrides: list[int] = [2, 2],  # Downsample stride
        patch_size=16,  # Patch size for what each feature pixel predicts in the future
        use_decoder_block: bool = False,  # Whether to use decoder block or not
        use_upsampling: bool = False,  # Whether to use bilinear upsample or not
        upsampling_dims: int = 32,  # Output conv hidden dimensions
        return_logits: bool = False,  # Whether to return the logits or not
        return_feat: bool = False,  # Whether to return the key intermediate features or not
        return_loss: bool = False,  # Whether to return the loss or not
        loss_fn: str = "VoxelFocalLoss",  # Loss function to use
        variable_mode: bool = True,  # Whether to use variable mode or not
        device: str = "cuda",
    ):
        super().__init__()
        assert (
            len(dims)
            == len(convkernels)
            == len(convdepths)
            == len(dskernels)
            == len(dsstrides)
        ), "Length of dims, convkernels, and convdepths should be the same."
        self._nstages = len(dims)

        self.frame_sizes = frame_sizes
        self.w, self.h = frame_sizes[:2]

        self.T = T
        self.dims = dims
        self.convkernels = convkernels
        self.convdepths = convdepths
        self.convbtlncks = convbtlncks
        self.convdilations = convdilations
        self.dskernels = dskernels
        self.dsstrides = dsstrides

        self.use_upsampling = use_upsampling
        self.upsampling_dims = upsampling_dims

        self.patch_size = patch_size
        self.multi_hash_encoder_args = copy(multi_hash_encoder)
        self.feature_size = dims[-1] if not use_upsampling else upsampling_dims

        self.use_decoder_block = use_decoder_block
        self.variable_mode = variable_mode
        self.return_logits = return_logits
        self.return_feat = return_feat
        self.return_loss = return_loss

        # Loss function
        if return_loss:
            from .utils import LOSSES

            self.loss_fn = LOSSES[loss_fn]

        self.downsample = torch.prod(torch.tensor(dsstrides)).item()
        assert self.use_upsampling or (self.downsample == self.patch_size), (
            "Downsample should be equal to the patch size."
        )

        multi_hash_encoder = copy(multi_hash_encoder)
        gp_resolution = (
            torch.log(
                torch.tensor(
                    multi_hash_encoder["finest_resolution"], dtype=torch.float32
                )
            )
            - torch.log(
                torch.tensor(
                    multi_hash_encoder["coarsest_resolution"], dtype=torch.float32
                )
            )
        ) / (multi_hash_encoder["levels"] - 1)

        resolutions = (
            torch.exp(
                torch.log(
                    torch.tensor(
                        multi_hash_encoder["coarsest_resolution"], dtype=torch.float32
                    )
                ).unsqueeze(0)
                + torch.arange(multi_hash_encoder["levels"]).unsqueeze(1)
                * gp_resolution.unsqueeze(0)
            )
            .type(torch.int32)
            .to(device)
        )  # since data minibatches are always on GPU

        multi_hash_encoder["resolutions"] = resolutions
        multi_hash_encoder["L_H"] = (
            (
                torch.log2(resolutions[:, :3] + 1).sum(dim=1)
                > multi_hash_encoder["log2_entries_per_level"]
            )
            .sum()
            .item()
        )
        multi_hash_encoder["L_NH"] = (
            multi_hash_encoder["levels"] - multi_hash_encoder["L_H"]
        )

        # Multi-resolution hash encoder for the feature field generating events.
        logger.info(
            "Instantiating Multi-resolution Hash Encoder with the following resolutions: "
            + f"{torch.Tensor(frame_sizes[:3]) / multi_hash_encoder['resolutions'][:, :3].cpu()}"
        )
        self.multi_hash_encoder = MultiResolutionHashEncoder(
            compile=compile, **multi_hash_encoder
        )

        in_channels = multi_hash_encoder["feature_size"] * multi_hash_encoder["levels"]

        kernel_strides_dilations = []
        for i in range(self._nstages):
            kernel_strides_dilations.append((dskernels[i], dsstrides[i], 1))
            kernel_strides_dilations.extend(
                [(convkernels[i], 1, convdilations[i]) for _ in range(convdepths[i])]
            )
        spatial_receptive_field = calculate_receptive_field(kernel_strides_dilations)
        logger.info(f"Spatial receptive field: {spatial_receptive_field}")

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    dims[0],
                    kernel_size=dskernels[0],
                    stride=dsstrides[0],
                    padding=(dskernels[0] - 1) // 2,
                ),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            )
        )
        for i in range(1, self._nstages):
            self.downsample_layers.append(
                nn.Sequential(
                    LayerNorm(dims[i - 1], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(
                        dims[i - 1],
                        dims[i],
                        kernel_size=dskernels[i],
                        stride=dsstrides[i],
                        padding=(dskernels[i] - 1) // 2,
                    ),
                )
            )

        self.stages = nn.ModuleList()
        for i in range(self._nstages):
            self.stages.append(
                nn.Sequential(
                    *[
                        Block(
                            dim=dims[i],
                            kernel_size=convkernels[i],
                            bottleneck=convbtlncks[i],
                            dilation=convdilations[i],
                        )
                        for _ in range(convdepths[i])
                    ]
                )
            )

        if use_upsampling:
            self.upsample_layers = nn.ModuleList()
            self.upsample_process = nn.ModuleList()

            for i in range(self._nstages - 1, 0, -1):
                # Upsample from dims[i] to dims[i-1], then concatenate with skip (dims[i-1])
                # Result will be 2*dims[i-1] channels after concatenation
                self.upsample_layers.append(
                    PixelShuffleUpsample(
                        dims[i], dims[i - 1], upscale_factor=dsstrides[i]
                    )
                )
                self.upsample_process.append(
                    nn.Sequential(
                        nn.GELU(),
                        nn.Conv2d(2 * dims[i - 1], dims[i - 1], kernel_size=1),
                        LayerNorm(dims[i - 1], eps=1e-6, data_format="channels_first"),
                        nn.GELU(),
                        nn.Conv2d(
                            dims[i - 1],
                            dims[i - 1],
                            kernel_size=3,
                            padding=1,
                            groups=dims[i - 1],
                        ),  # Depthwise conv
                        LayerNorm(dims[i - 1], eps=1e-6, data_format="channels_first"),
                    )
                )

            self.upsample_layers.append(
                PixelShuffleUpsample(
                    dims[0],
                    upsampling_dims,
                    upscale_factor=dsstrides[0] // self.patch_size,
                )
            )

            self.upsample_process.append(
                nn.Sequential(
                    nn.GELU(),
                    nn.Conv2d(2 * upsampling_dims, upsampling_dims, kernel_size=1),
                    LayerNorm(upsampling_dims, eps=1e-6, data_format="channels_first"),
                    nn.GELU(),
                    nn.Conv2d(
                        upsampling_dims,
                        upsampling_dims,
                        kernel_size=3,
                        padding=1,
                        groups=upsampling_dims,
                    ),
                    LayerNorm(upsampling_dims, eps=1e-6, data_format="channels_first"),
                )
            )

            self.downsample_hash_to_patchsize = nn.Conv2d(
                in_channels,
                upsampling_dims,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                padding=(self.patch_size - 1) // 2,
            )

        if use_decoder_block:
            self.decoder = Block(
                dim=dims[-1] if not use_upsampling else upsampling_dims, kernel_size=7
            )

        self.pred = nn.Conv2d(
            dims[-1] if not use_upsampling else upsampling_dims,
            patch_size**2 * T,
            kernel_size=1,
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        """
        x: (N, patch_size**2 * T, W/patch_size, H/patch_size)
        imgs: (N, W*patch_size, H*patch_size, T)
        """
        if self.patch_size == 1:
            return x.permute(0, 2, 3, 1)

        p = self.patch_size
        n, _, w, h = x.shape
        x = x.permute(0, 2, 3, 1).reshape(n, w, h, p, p, -1)
        imgs = x.permute(0, 1, 3, 2, 4, 5).reshape(n, w * p, h * p, -1)
        return imgs

    @classmethod
    def init_from_config(
        cls,
        eventff_config: str,
        return_logits: bool = False,
        return_feat: bool = False,
        return_loss: bool = False,
        loss_fn: str = "VoxelFocalLoss",
        **kwargs,
    ):
        with open(eventff_config, "r") as f:
            conf = yaml.safe_load(f)

        return cls(
            multi_hash_encoder=conf["multi_hash_encoder"],
            T=conf["T"],
            frame_sizes=conf["frame_sizes"],
            dims=conf["dims"],
            convkernels=conf["convkernels"],
            convdepths=conf["convdepths"],
            convbtlncks=conf["convbtlncks"],
            convdilations=conf["convdilations"],
            dskernels=conf["dskernels"],
            dsstrides=conf["dsstrides"],
            patch_size=conf["patch_size"],
            use_decoder_block=conf.get("use_decoder_block", False),
            use_upsampling=conf.get("use_upsampling", False),
            upsampling_dims=conf.get("upsampling_dims", [128, 32]),
            return_logits=return_logits,
            return_loss=return_loss,
            return_feat=return_feat,
            loss_fn=loss_fn,
            device=conf.get("device", "cuda"),
            variable_mode=conf.get("variable_mode", True),
        )

    def save_config(self, path: str):
        with open(path, "w") as f:
            yaml.dump(
                {
                    "model": "EventPatchFF",
                    "multi_hash_encoder": self.multi_hash_encoder_args,
                    "T": self.T,
                    "frame_sizes": self.frame_sizes,
                    "dims": self.dims,
                    "convkernels": self.convkernels,
                    "convdepths": self.convdepths,
                    "convbtlncks": self.convbtlncks,
                    "convdilations": self.convdilations,
                    "dskernels": self.dskernels,
                    "dsstrides": self.dsstrides,
                    "patch_size": self.patch_size,
                    "use_upsampling": self.use_upsampling,
                    "upsampling_dims": self.upsampling_dims,
                    "use_decoder_block": self.use_decoder_block,
                    "variable_mode": self.variable_mode,
                },
                f,
                default_flow_style=None,
            )

    def forward_fixed(self, currentBlock: torch.Tensor) -> torch.Tensor:
        """
        currentBlock: torch.Tensor (B,N,3/4)
        """
        B, N = currentBlock.shape[0], currentBlock.shape[1]
        curr_x = (currentBlock[:, :, 0] * self.w).round().int()
        curr_y = (currentBlock[:, :, 1] * self.h).round().int()

        encoded_events = self.multi_hash_encoder(
            currentBlock
        )  # (B,N,3)/(B,N,4) -> (B,N,L*F)

        feature_field = torch.zeros(
            (B, self.w, self.h, encoded_events.shape[-1]), device=encoded_events.device
        )
        batch_indices = (
            torch.arange(B).to(encoded_events.device).view(-1, 1).expand(B, N)
        )
        feature_field.index_put_(
            (batch_indices, curr_x, curr_y), encoded_events, accumulate=True
        )
        return feature_field

    def forward_variable(
        self, currentBlock: torch.Tensor, eventCounts: torch.Tensor
    ) -> torch.Tensor:
        """
        currentBlock: torch.Tensor (N, 3/4) N = N1 + N2 + N3 + ... + NB
        eventCounts: torch.Tensor (B,)    [N1, N2, N3, ..., NB]
        """
        B = eventCounts.shape[0]
        curr_x = (currentBlock[:, 0] * self.w).round().int()
        curr_y = (currentBlock[:, 1] * self.h).round().int()

        encoded_events = (
            self.multi_hash_encoder(currentBlock.unsqueeze(0)).clone().squeeze(0)
        )  # (N,3)/(N,4) -> (N,L*F)

        feature_field = torch.zeros(
            (B, self.w, self.h, encoded_events.shape[-1]), device=encoded_events.device
        )
        batch_indices = torch.repeat_interleave(
            eventCounts
        ).int()  # Offending line, cant compile. I am not spending time on this for now.
        feature_field.index_put_(
            (batch_indices, curr_x, curr_y), encoded_events, accumulate=True
        )
        return feature_field.permute(0, 3, 1, 2)  # (B, C, W, H)

    def forward_loss(
        self, logits: torch.Tensor, futureBlock: torch.Tensor, valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        logits: torch.Tensor (B,W,H,T)
        futureBlock: torch.Tensor (B,W,H,T)
        """
        loss = self.loss_fn(logits, futureBlock, valid_mask)
        return loss

    def forward(
        self,
        currentBlock: torch.Tensor,
        eventCounts: torch.Tensor = None,
        futureBlock: torch.Tensor = None,
        valid_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        currentBlock: torch.Tensor (B, N, 3/4) or (N, 3/4) N = N1 + N2 + N3 + ... + NB
        eventCounts: torch.Tensor (B,) [N1, N2, N3, ..., NB]
        futureBlock: torch.Tensor (B, W, H, T)
        valid_mask: torch.Tensor (B, W, H)
        """
        if not self.return_logits and not self.return_feat and not self.return_loss:
            return None

        if self.variable_mode:
            x = self.forward_variable(currentBlock, eventCounts)  # (B,C,W,H)
        else:
            x = self.forward_fixed(currentBlock)  # (B,C,W,H)

        skip_connections = []  # Store the input as the first skip connection
        if self.use_upsampling:
            skip_connections.append(x)

        for i in range(self._nstages):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            if self.use_upsampling and i < self._nstages - 1:
                skip_connections.append(x)

        if self.use_upsampling:
            for i in range(len(self.upsample_layers) - 1):
                skip = skip_connections[
                    -(i + 1)
                ]  # Get skip connections in reverse order
                x = self.upsample_layers[i](x)  # Upsample
                x = torch.cat([skip, x], dim=1)  # Concatenate with skip connection
                x = self.upsample_process[i](x)  # Process concatenated features

            x = self.upsample_layers[-1](x)
            downsampled_hash = self.downsample_hash_to_patchsize(
                skip_connections[0]
            )  # (B, C, W/ps, H/ps)
            x = torch.cat(
                [x, downsampled_hash], dim=1
            )  # Concatenate along channel dimension
            x = self.upsample_process[-1](x)

        if self.return_feat:
            feat = x.permute(0, 2, 3, 1)

        if not self.return_logits and self.return_feat:
            return None, feat

        if self.use_decoder_block:
            x = self.decoder(x)
        logits = self.pred(x)  # (B,C,W,H) -> (B,patch_size**2*T,W,H)

        logits = self.unpatchify(logits)  # (B,patch_size**2*T,W/ps,H/ps) -> (B,W,H,T)
        if self.return_loss:
            loss = self.forward_loss(logits, futureBlock, valid_mask)

        ret = (None,)
        if self.return_logits:
            ret = (logits,)
        if self.return_feat:
            ret += (feat,)
        if self.return_loss:
            ret += (loss,)
        return ret


def load_weights_ckpt(model: nn.Module, eventff_ckpt: str, strict: bool = True):
    """Load model weights from checkpoint."""
    ckpt = torch.load(eventff_ckpt, weights_only=True)
    model.load_state_dict(ckpt["model"], strict=strict)
    torch.cuda.empty_cache()
    return ckpt.get("epoch", 0), ckpt.get("loss", 0), ckpt.get("acc", 0)


def init_event_model(
    eventff_config: str,
    return_logits: bool = True,
    return_feat: bool = False,
    return_loss: bool = False,
    loss_fn: str = "VoxelFocalLoss",
    **kwargs,
) -> nn.Module:
    with open(eventff_config, "r") as f:
        conf = yaml.safe_load(f)
        model = conf["model"]
    if model == "EventPatchFF":
        return EventPatchFF.init_from_config(
            eventff_config,
            return_logits=return_logits,
            return_feat=return_feat,
            return_loss=return_loss,
            loss_fn=loss_fn,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model: {model}")


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_receptive_field(layers):
    """
    Calculate the receptive field for a sequence of layers.

    Parameters:
        layers: List of tuples [(kernel_size1, stride1, dilation1), (kernel_size2, stride2, dilation2), ...]

    Returns:
        Receptive field size at the final layer.
    """
    receptive_field = 1
    product = 1
    for kernel_size, stride, dilation in layers:
        product *= stride
        receptive_field += (kernel_size - 1) * dilation * product
    return receptive_field
