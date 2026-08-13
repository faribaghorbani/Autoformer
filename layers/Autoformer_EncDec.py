import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class adaptive_multi_scale_series_decomp(nn.Module):
    """
    Adaptive Multi-Scale Series Decomposition.

    Instead of using one fixed moving-average window, this module
    computes several moving averages at different temporal scales
    and learns sample-specific, channel-specific weights to combine them.

    Input:
        x: [B, L, C]

    Output:
        seasonal: [B, L, C]
        trend:    [B, L, C]
    """

    def __init__(
        self,
        channels,
        kernel_sizes=(13, 25, 49),
        hidden_dim=None,
    ):
        super(adaptive_multi_scale_series_decomp, self).__init__()

        if len(kernel_sizes) < 2:
            raise ValueError(
                "Adaptive multi-scale decomposition requires at least "
                "two kernel sizes."
            )

        # Moving-average kernels must be odd so that the temporal
        # length remains unchanged after symmetric padding.
        for kernel_size in kernel_sizes:
            if kernel_size % 2 == 0:
                raise ValueError(
                    f"Kernel size {kernel_size} is even. "
                    "All kernel sizes must be odd."
                )

            if kernel_size < 3:
                raise ValueError(
                    f"Kernel size {kernel_size} is too small. "
                    "Use odd values >= 3."
                )

        self.channels = channels
        self.kernel_sizes = tuple(kernel_sizes)
        self.num_scales = len(kernel_sizes)

        # Default hidden size for the gating network.
        if hidden_dim is None:
            hidden_dim = max(channels // 4, 16)

        self.hidden_dim = hidden_dim

        # One fixed moving-average operator for each temporal scale.
        self.moving_avgs = nn.ModuleList([
            moving_avg(kernel_size, stride=1)
            for kernel_size in self.kernel_sizes
        ])

        # ------------------------------------------------------------------
        # Adaptive scale-selection network
        #
        # Input:
        #     global channel representation -> [B, C]
        #
        # Output:
        #     one weight for every (channel, scale) pair -> [B, C, K]
        # ------------------------------------------------------------------

        self.scale_gate = nn.Sequential(
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels * self.num_scales)
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, L, C]

        Returns:
            seasonal: Tensor [B, L, C]
            trend:    Tensor [B, L, C]
        """

        if x.dim() != 3:
            raise ValueError(
                f"Expected input with shape [B, L, C], got {x.shape}"
            )

        batch_size, seq_len, channels = x.shape

        if channels != self.channels:
            raise ValueError(
                f"Adaptive decomposition was created for {self.channels} "
                f"channels, but received input with {channels} channels."
            )

        # --------------------------------------------------------------
        # 1. Compute multiple smoothed versions.
        # --------------------------------------------------------------
        multi_scale_trends = []

        for moving_avg_layer in self.moving_avgs:
            trend_k = moving_avg_layer(x)
            multi_scale_trends.append(trend_k)

        # [B, L, C, K]
        multi_scale_trends = torch.stack(
            multi_scale_trends,
            dim=-1
        )

        # --------------------------------------------------------------
        # 2. Build an input-dependent scale representation.
        #
        # Mean over time:
        #
        #       [B, L, C] -> [B, C]
        #
        # This lets the gate understand the current state of each
        # feature/channel.
        # --------------------------------------------------------------
        summary = x.mean(dim=1)

        # [B, C*K]
        gate_logits = self.scale_gate(summary)

        # [B, C, K]
        gate_logits = gate_logits.view(
            batch_size,
            channels,
            self.num_scales
        )

        # Normalize across temporal scales.
        #
        # For each sample and channel:
        #
        #     sum_k alpha[b,c,k] = 1
        #
        scale_weights = torch.softmax(
            gate_logits,
            dim=-1
        )

        # [B, 1, C, K]
        scale_weights = scale_weights.unsqueeze(1)

        # --------------------------------------------------------------
        # 3. Adaptive weighted fusion.
        #
        # trend =
        #     alpha_1 * MA_13(x)
        #   + alpha_2 * MA_25(x)
        #   + alpha_3 * MA_49(x)
        # --------------------------------------------------------------
        trend = torch.sum(
            multi_scale_trends * scale_weights,
            dim=-1
        )

        # --------------------------------------------------------------
        # 4. Seasonal residual.
        # --------------------------------------------------------------
        seasonal = x - trend

        return seasonal, trend


def build_series_decomp(
    decomp_type,
    channels,
    moving_avg_kernel=25,
    decomp_kernels=(13, 25, 49),
):
    """
    Construct either the original fixed-scale decomposition
    or the new adaptive multi-scale decomposition.

    Args:
        decomp_type:
            "fixed"    -> original Autoformer decomposition
            "adaptive" -> adaptive multi-scale decomposition

        channels:
            Number of input channels.

        moving_avg_kernel:
            Original Autoformer moving-average kernel.

        decomp_kernels:
            Kernels used by adaptive decomposition.

    Returns:
        nn.Module implementing series decomposition.
    """

    if decomp_type == "fixed":
        return series_decomp(moving_avg_kernel)

    elif decomp_type == "adaptive":
        return adaptive_multi_scale_series_decomp(
            channels=channels,
            kernel_sizes=decomp_kernels,
        )

    else:
        raise ValueError(
            f"Unknown decomp_type='{decomp_type}'. "
            "Expected one of: ['fixed', 'adaptive']"
        )


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(
        self,
        attention,
        d_model,
        d_ff=None,
        moving_avg=25,
        dropout=0.1,
        activation="relu",
        decomp_type="fixed",
        decomp_kernels=(13, 25, 49),
    ):
        super(EncoderLayer, self).__init__()

        d_ff = d_ff or 4 * d_model

        self.attention = attention

        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_ff,
            kernel_size=1,
            bias=False
        )

        self.conv2 = nn.Conv1d(
            in_channels=d_ff,
            out_channels=d_model,
            kernel_size=1,
            bias=False
        )

        # --------------------------------------------------------------
        # Adaptive or fixed decomposition
        # --------------------------------------------------------------
        self.decomp1 = build_series_decomp(
            decomp_type=decomp_type,
            channels=d_model,
            moving_avg_kernel=moving_avg,
            decomp_kernels=decomp_kernels,
        )

        self.decomp2 = build_series_decomp(
            decomp_type=decomp_type,
            channels=d_model,
            moving_avg_kernel=moving_avg,
            decomp_kernels=decomp_kernels,
        )

        self.dropout = nn.Dropout(dropout)

        self.activation = (
            F.relu
            if activation == "relu"
            else F.gelu
        )
    

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y)
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        c_out,
        d_ff=None,
        moving_avg=25,
        dropout=0.1,
        activation="relu",
        decomp_type="fixed",
        decomp_kernels=(13, 25, 49),
    ):
        super(DecoderLayer, self).__init__()

        d_ff = d_ff or 4 * d_model

        self.self_attention = self_attention
        self.cross_attention = cross_attention

        self.conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_ff,
            kernel_size=1,
            bias=False
        )

        self.conv2 = nn.Conv1d(
            in_channels=d_ff,
            out_channels=d_model,
            kernel_size=1,
            bias=False
        )

        # --------------------------------------------------------------
        # Adaptive or fixed decomposition
        # --------------------------------------------------------------
        self.decomp1 = build_series_decomp(
            decomp_type=decomp_type,
            channels=d_model,
            moving_avg_kernel=moving_avg,
            decomp_kernels=decomp_kernels,
        )

        self.decomp2 = build_series_decomp(
            decomp_type=decomp_type,
            channels=d_model,
            moving_avg_kernel=moving_avg,
            decomp_kernels=decomp_kernels,
        )

        self.decomp3 = build_series_decomp(
            decomp_type=decomp_type,
            channels=d_model,
            moving_avg_kernel=moving_avg,
            decomp_kernels=decomp_kernels,
        )

        self.dropout = nn.Dropout(dropout)

        self.projection = nn.Conv1d(
            in_channels=d_model,
            out_channels=c_out,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode='circular',
            bias=False
        )

        self.activation = (
            F.relu
            if activation == "relu"
            else F.gelu
        )


    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
