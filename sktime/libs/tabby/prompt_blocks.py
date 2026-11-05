# ruff: noqa
"""Prompt building blocks shared by the prompt-tuning wrappers.

These four modules turn a raw context window into a small set of prompt vectors:

    SegmentedStatsExtractor -> per-segment summary statistics of the context
    Chronos2StatsExtractor  -> whole-window summary statistics (non-segmented path)
    SegmentRefiner          -> cross-attention that lets prompt slots read segments
    ContextPromptGenerator  -> low-rank map from statistics to a prompt delta

They are backbone-agnostic: nothing here touches PatchTST-FM specifics.
"""

from __future__ import annotations

import math

from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
nn = _safe_import("torch.nn")


class Chronos2StatsExtractor(nn.Module):
    def __init__(self, n_stats: int = 10):
        super().__init__()
        self.n_stats = n_stats

    def _autocorr(self, centered, valid, lag):
        if centered.shape[1] <= lag:
            return torch.zeros(centered.shape[0], device=centered.device)
        x0, x1 = centered[:, :-lag], centered[:, lag:]
        pair_valid = valid[:, :-lag] * valid[:, lag:]
        n = pair_valid.sum(-1).clamp(min=1)
        cov = (x0 * x1 * pair_valid).sum(-1) / n
        var = (centered**2 * valid).sum(-1) / valid.sum(-1).clamp(min=1)
        return (cov / var.clamp(min=1e-6)).clamp(-1, 1)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        B, T = values.shape
        v = values * mask
        n_valid = mask.sum(-1).clamp(min=1)
        stats = []

        mean = v.sum(-1) / n_valid
        stats.append(mean)

        var = ((v - mean.unsqueeze(-1)) ** 2 * mask).sum(-1) / n_valid.clamp(min=2)
        stats.append(var.sqrt().clamp(max=10.0))

        v_min = values.clone()
        v_min[mask == 0] = 1e6
        stats.append(v_min.min(-1).values.clamp(-10, 10))

        v_max = values.clone()
        v_max[mask == 0] = -1e6
        stats.append(v_max.max(-1).values.clamp(-10, 10))

        half = T // 2
        m1 = (v[:, :half] * mask[:, :half]).sum(-1) / mask[:, :half].sum(-1).clamp(
            min=1
        )
        m2 = (v[:, half:] * mask[:, half:]).sum(-1) / mask[:, half:].sum(-1).clamp(
            min=1
        )
        stats.append((m2 - m1).clamp(-5, 5))

        diff = v[:, 1:] - v[:, :-1]
        dv = mask[:, 1:] * mask[:, :-1]
        nd = dv.sum(-1).clamp(min=1)
        dm = (diff * dv).sum(-1) / nd
        dvar = ((diff * dv - dm.unsqueeze(-1)) ** 2 * dv).sum(-1) / nd.clamp(min=2)
        stats.append(dm.clamp(-5, 5))
        stats.append(dvar.sqrt().clamp(max=10.0))

        centered = (v - mean.unsqueeze(-1)) * mask
        stats.append(self._autocorr(centered, mask, 1))
        stats.append(n_valid / T)
        stats.append(torch.log1p(n_valid) / 10.0)

        result = torch.stack(stats[: self.n_stats], dim=-1)
        return torch.nan_to_num(result, nan=0.0)


class SegmentedStatsExtractor(nn.Module):
    """
    context [B, T], mask [B, T]  ->  seg_stats [B, S, n_stats], seg_valid [B, S]

    The valid region is RIGHT-ALIGNED (real data on the right, left is padding); the
    valid data's right edge is always at index T (the runtime tensor width), in BOTH
    training (T == context_length via InstanceSplitter past_length) and test (T == the
    batch's longest series). We anchor segments at that right edge and walk leftward,
    placing the MOST-RECENT segment at slot S-1 (slot index increases with recency).

    CRITICAL — train/test consistency:
      seg_len is anchored to the CONFIGURED context_length (a constant W = context_length
      // S), NEVER to the runtime tensor width T. At test the context tensor is only as
      wide as the batch's longest series, so T // S would shrink seg_len and split a given
      series completely differently than at training. Tying seg_len to the constant W (and
      to each series' own valid_len) makes a given series split identically at train and
      test, which is what the trained seg_pos / refiner require.

    seg_mode:
      fixed_len        (A): seg_len = W (= context_length // S, constant). A saturated
                            context -> S full segments == _m; short -> fewer segments,
                            covering exactly the valid data, degenerating toward mode=none.
      adaptive_horizon (C): seg_len = max(floor, valid_len // S),
                            floor = clamp(season_or_pred, min_seg_len, W).
                            W-cap => floor never exceeds the long width => a saturated
                            context gives C == A == _m; short -> ~one cycle per segment.
    """

    def __init__(
        self,
        n_stats=10,
        max_segments=16,
        seg_mode="fixed_len",
        seg_fixed_len=256,
        seg_len_k=2,
        min_seg_len=48,
        prediction_length=48,
        season_period=1,
        context_length=4096,
    ):
        super().__init__()
        self.n_stats = n_stats
        self.max_segments = max_segments
        self.seg_mode = seg_mode
        self.seg_fixed_len = seg_fixed_len
        self.seg_len_k = seg_len_k  # kept for config compat; unused by C
        self.min_seg_len = min_seg_len
        self.prediction_length = prediction_length
        self.season_period = season_period
        self.context_length = context_length  # configured padded width (e.g. 4096); the
        # constant reference for seg_len, NOT runtime T
        self.base = Chronos2StatsExtractor(n_stats)

    def _seg_len_for(self, valid_len, W):
        # W = context_length // S  (constant long-segment width, e.g. 256)
        if self.seg_mode == "adaptive_horizon":
            base = (
                self.season_period
                if (self.season_period and self.season_period > 1)
                else self.prediction_length
            )
            floor = min(
                W, max(self.min_seg_len, int(base))
            )  # >= ~one cycle, capped at W
            return int(max(floor, valid_len // self.max_segments))
        # fixed_len (A): constant 256-wide segments
        return int(W)

    def forward(self, values, mask):
        B, T = values.shape
        S = self.max_segments
        device, dtype = values.device, values.dtype

        # Constant reference width — anchored to context_length, NOT runtime T.
        W = max(1, self.context_length // S)

        seg_stats = torch.zeros(B, S, self.n_stats, device=device, dtype=dtype)
        seg_valid = torch.zeros(B, S, device=device, dtype=dtype)
        valid_lens = mask.sum(-1).long().tolist()  # single host sync

        index_map = []  # (b, slot, start, end)
        for b in range(B):
            vlen = valid_lens[b]
            if vlen <= 0:
                continue
            seg_len = min(self._seg_len_for(vlen, W), vlen)
            if seg_len <= 0:
                continue
            if self.seg_mode == "adaptive_horizon":
                n = min(S, max(1, vlen // seg_len))  # floor: every seg >= floor
            else:
                n = min(
                    S, max(1, (vlen + seg_len - 1) // seg_len)
                )  # A: ceil -> cover all valid (== _m on long)
            right = T  # valid data's right edge (== T, right-aligned, train & test)
            left_valid = T - vlen
            for j in range(n):  # j = 0 -> most recent
                end = right - j * seg_len
                start = max(left_valid, end - seg_len)
                if end <= start:
                    break
                slot = S - 1 - j  # newest -> last slot (back placement)
                index_map.append((b, slot, start, end))

        if not index_map:
            return seg_stats, seg_valid

        # BUGFIX (batch-dependent stats): stats are computed PER LENGTH GROUP so that
        # each segment's stats use its OWN length as T. Packing all segments into one
        # [N, max_len] buffer (old code) made length-relative stats wrong for any
        # segment shorter than the batch max:
        #   - trend (half-split at T//2): a short left-aligned segment had its whole
        #     data in the first half -> m2=0 -> trend = -seg_mean (trend info destroyed)
        #   - fraction_observed (n_valid / T): diluted by the buffer padding
        # and, worse, max_len depends on the BATCH composition -> the same series got
        # different seg stats at train (mixed-dataset batches) vs test (per-dataset
        # batches) — a silent train/test mismatch. Grouping by identical L removes the
        # batch dependence entirely; there are only a handful of unique lengths per
        # batch, so the overhead is negligible. Long saturated contexts (all L == W)
        # form a single group and reproduce the previous behaviour exactly.
        groups: Dict[int, list] = {}
        for item in index_map:
            b, slot, s, e = item
            groups.setdefault(e - s, []).append(item)

        for L, items in groups.items():
            vals = torch.stack([values[b, s:e] for (b, _, s, e) in items])  # [G, L]
            msk = torch.stack([mask[b, s:e] for (b, _, s, e) in items])  # [G, L]
            st = self.base(vals, msk)  # [G, n_stats]
            has_data = (msk.sum(-1) > 0).to(dtype)  # [G]
            for i, (b, slot, s, e) in enumerate(items):
                seg_stats[b, slot] = st[i]
                # BUGFIX: a segment inside the valid span can still be fully missing
                # (e.g. kdd-style NaN gaps). Old code set seg_valid=1 unconditionally,
                # letting attention read an all-zero stats vector. Follow _m: valid
                # only if the segment contains >0 observed points.
                seg_valid[b, slot] = has_data[i]
        return seg_stats, seg_valid


class SegmentRefiner(nn.Module):
    def __init__(
        self,
        n_stats,
        d_model,
        prompt_len,
        n_segments,
        d_attn=64,
        n_heads=4,
        query_from_base=True,
        relative_pos=True,
    ):
        super().__init__()
        assert d_attn % n_heads == 0
        self.query_from_base = query_from_base
        self.relative_pos = relative_pos

        self.kv_encoder = nn.Sequential(
            nn.Linear(n_stats, d_attn),
            nn.GELU(),
            nn.Linear(d_attn, d_attn),
        )
        self.seg_pos = nn.Parameter(torch.zeros(n_segments, d_attn))  # temporal order
        nn.init.normal_(self.seg_pos, std=0.02)

        if query_from_base:
            self.q_from_base = nn.Linear(d_model, d_attn)  # base_prompt → query
        else:
            self.queries = nn.Parameter(torch.zeros(prompt_len, d_attn))
            nn.init.normal_(self.queries, std=0.02)

        self.k_proj = nn.Linear(d_attn, d_attn)
        self.v_proj = nn.Linear(d_attn, d_attn)

        self.out_proj = nn.Linear(d_attn, d_model)
        nn.init.zeros_(self.out_proj.weight)  # zero-init → refine≈0 at start
        nn.init.zeros_(self.out_proj.bias)

        self.n_heads = n_heads
        self.head_dim = d_attn // n_heads
        self.d_attn = d_attn

    def _relative_pos(self, seg_valid):
        """
        Map each sample's n_valid segments onto the learned seg_pos grid by RELATIVE
        position. Valid segments live in the BACK slots in chronological order (ascending
        slot index = oldest -> newest), so we fill the ACTUAL valid slots with
        linspace(0,1,nv) interpolated over the grid: oldest -> grid[0] ("start"),
        newest -> grid[S-1] ("end"). A 5-segment and a 16-segment series therefore share
        start->end semantics. Returns [B, S, d_attn].
        """
        B, S = seg_valid.shape
        device = seg_valid.device
        grid = self.seg_pos  # [G, d_attn] (G = refiner n_segments)
        # Index the grid by G (its own row count), never by the number of input slots S.
        # An earlier form used rel*(S-1), which silently assumed S == G and went out of
        # bounds when a long branch with more segments shared a narrower refiner grid.
        G = grid.shape[0]
        out = torch.zeros(B, S, grid.shape[-1], device=device, dtype=grid.dtype)
        for b in range(B):
            valid_idx = torch.nonzero(seg_valid[b] > 0, as_tuple=False).squeeze(
                -1
            )  # ascending = old->new
            nv = int(valid_idx.numel())
            if nv == 0:
                continue
            if nv == 1:
                out[b, valid_idx[0]] = grid[-1]  # single segment = most-recent end
                continue
            rel = torch.linspace(0, 1, nv, device=device)  # [nv]  old->new
            idx = rel * (G - 1)
            lo = idx.floor().long().clamp(0, G - 1)
            hi = idx.ceil().long().clamp(0, G - 1)
            w = (idx - lo.float()).unsqueeze(-1)
            interp = (1 - w) * grid[lo] + w * grid[hi]  # [nv, d_attn]
            out[b, valid_idx] = interp
        return out

    def forward(self, seg_matrix, seg_valid, base_prompt):
        # seg_matrix: [B, S, n_stats], seg_valid: [B, S], base_prompt: [B, M, d_model]
        B, S, _ = seg_matrix.shape
        M = base_prompt.shape[1]

        if self.relative_pos:
            pos = self._relative_pos(seg_valid)  # [B, S, d_attn]
            kv = self.kv_encoder(seg_matrix) + pos
        else:
            kv = self.kv_encoder(seg_matrix) + self.seg_pos[:S].unsqueeze(0)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        if self.query_from_base:
            q = self.q_from_base(base_prompt)  # [B, M, d_attn]
        else:
            q = self.queries.unsqueeze(0).expand(B, M, -1)

        q = q.view(B, M, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(seg_valid.view(B, 1, 1, S) == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # guard: all-invalid row

        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, M, self.d_attn)
        out = self.out_proj(out)  # [B, M, d_model]
        # Skip refinement per-sample when there is <=1 valid segment: a single segment
        # spans ~all the valid data, so its stats duplicate the global stats already in
        # base_prompt -> refine is meaningless. Zero those rows so they fall back to pure
        # base_prompt (== none) for that sample, without affecting multi-segment samples.
        keep = ((seg_valid > 0).sum(-1) > 1).to(out.dtype).view(B, 1, 1)
        return out * keep  # [B, M, d_model]


class Chronos2ContextPromptGenerator(nn.Module):
    # basis_init_std defaults to 0.001, the value every reported experiment used.
    # basis_init_scale exists only for diagnostics: the injected term is
    # sigma(ctx_gate) * ctx_prompt, so the gate and the basis norm are multiplicatively
    # degenerate; rescaling the basis at fixed gate separates the two factors.
    def __init__(
        self,
        feat_dim,
        prompt_len,
        d_model,
        rank=8,
        hidden_dim=64,
        n_layers=2,
        basis_init_std=0.001,
    ):
        super().__init__()
        self.rank = rank
        layers = [nn.Linear(feat_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, rank))
        self.mlp = nn.Sequential(*layers)
        self.basis = nn.Parameter(torch.zeros(rank, prompt_len, d_model))
        nn.init.normal_(self.basis, std=basis_init_std)

    def forward(self, features):
        weights = self.mlp(features)
        return torch.einsum("br, rpd -> bpd", weights, self.basis)
