import torch
import torch.nn as nn
import math
# add dependency
import torch.nn.functional as F


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:
    (1) period-based dependencies discovery
    (2) time delay aggregation
    This block can replace the self-attention family mechanism seamlessly.
    """
    # experiment 2A changes
    def __init__(
    self,
    mask_flag=True,
    factor=1,
    scale=None,
    attention_dropout=0.1,
    output_attention=False,
    period_mode='original',
    router_hidden=16,
    router_bins=8
    ):
        super(AutoCorrelation, self).__init__()

        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.period_mode = period_mode
        self.router_hidden = router_hidden
        self.router_bins = router_bins

        if self.period_mode == 'router':
            self.period_router = nn.Sequential(
                nn.Linear(router_bins + 2, router_hidden),
                nn.GELU(),
                nn.Linear(router_hidden, 1)
            )

            nn.init.zeros_(self.period_router[-1].weight)
            nn.init.zeros_(self.period_router[-1].bias)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg


    def time_delay_agg_samplewise(self, values, corr):
        """
        Experiment 2A:
        Sample-wise lag selection during both training and inference.

        values: [B, H, E, L]
        corr:   [B, H, E, L]

        Selected delays:
            [B, K]

        Unlike original Autoformer training, we do NOT average
        the autocorrelation across the batch before Top-K
        """

        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]

        # Number of selected periods
        top_k = int(self.factor * math.log(length))
        top_k = max(1, min(top_k, length))

        # --------------------------------------------------
        # [B,H,E,L] -> [B,L]
        #
        # We keep the original speedup idea of averaging
        # heads and latent channels.
        #
        # But importantly, batch B is preserved.
        # --------------------------------------------------
        mean_value = torch.mean(
            torch.mean(corr, dim=1),
            dim=1
        )

        # --------------------------------------------------
        # Per-sample Top-K
        #
        # weights: [B,K]
        # delay:   [B,K]
        # --------------------------------------------------
        weights, delay = torch.topk(
            mean_value,
            top_k,
            dim=-1
        )

        # Original Autoformer weighting rule:
        # higher autocorrelation -> larger aggregation weight
        tmp_corr = torch.softmax(weights, dim=-1)

        # [1,1,1,L]
        init_index = torch.arange(
            length,
            device=values.device
        ).view(1, 1, 1, length)

        # Allows circular delayed gathering
        # [B,H,E,L] -> [B,H,E,2L]
        tmp_values = values.repeat(1, 1, 1, 2)

        delays_agg = torch.zeros_like(values).float()

        for i in range(top_k):

            # delay[:, i]:
            # [B]
            #
            # -> [B,1,1,1]
            current_delay = delay[:, i].view(
                batch, 1, 1, 1
            )

            # [B,1,1,L]
            tmp_delay = init_index + current_delay

            # Expand for all heads/channels
            tmp_delay = tmp_delay.expand(
                batch,
                head,
                channel,
                length
            )

            # Retrieve the shifted V sequence
            pattern = torch.gather(
                tmp_values,
                dim=-1,
                index=tmp_delay
            )

            # Sample-specific weight
            current_weight = tmp_corr[:, i].view(
                batch, 1, 1, 1
            )

            delays_agg = (
                delays_agg
                + pattern * current_weight
            )

        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        if L > S:
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim=-1)

        # time delay agg
        if self.training:
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
