"""Cross Entropy Method solver for model-based planning."""

import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from loguru import logger as logging

from .solver import Costable


class CEMSolver:
    """Cross Entropy Method solver for action optimization.

    Args:
        model: World model implementing the Costable protocol.
        batch_size: Number of environments to process in parallel.
        num_samples: Number of action candidates to sample per iteration.
        var_scale: Initial variance scale for the action distribution.
        n_steps: Number of CEM iterations.
        topk: Number of elite samples to keep for distribution update.
        device: Device for tensor computations.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        model: Costable,
        batch_size: int = 1,
        num_samples: int = 300,
        var_scale: float = 1,
        n_steps: int = 30,
        topk: int = 30,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        action_clamp: float | list | tuple | torch.Tensor | None = None,
        binary_gripper: bool = False,
        env_clip_range: tuple[float, float] | list | None = None,
        projection_mode: str = "cost_only",
    ) -> None:
        """
        Args:
            action_clamp: If not None, clamp all sampled action candidates AND the
                CEM mean after each iteration in NORMALIZED action space. Accepts:
                  - scalar (float): symmetric [-c, +c] on every physical dim.
                  - list/tuple/Tensor of length physical_dim: per-dim symmetric
                    bound. Broadcast across action_block slices so each
                    physical-dim bound applies identically within every block.
                Prevents CEM from drifting into OOD regions where the predictor
                extrapolates (predictor trained on N(0, 1)-ish normalized
                actions; |norm| >> 1 is extrapolation).

                Per-dim is useful when the env has bounds that differ per dim.
                Compute each entry from the downstream StandardScaler as
                ``B_i = max((1 - mean_i)/std_i, (1 + mean_i)/std_i)``.

                Set to None (default) to disable.
            binary_gripper: If True, snap the LAST physical action dim (gripper)
                to sign(x) at every sampled candidate and CEM mean update.
                Prevents CEM from picking intermediate gripper values (which can
                cause partial release after grasp). Default False preserves the
                continuous-gripper behavior.
            env_clip_range: (low, high) env-space bounds. Activates env-space
                projection (denorm → clip(low, high) → renorm) when combined
                with an action_scaler injected via :meth:`set_action_scaler`.
                See :attr:`projection_mode` for how the projection is applied.
                ``None`` (default) disables projection entirely.
            projection_mode: How the env-space projection interacts with CEM
                state. Accepts:
                  - ``"cost_only"`` (default): project a COPY of candidates
                    for cost eval only; CEM's ``candidates`` tensor used for
                    elite selection + mean update stays on the pre-projection
                    values. WM sees in-distribution actions while CEM mean is
                    free to drift past env boundaries.
                  - ``"state"``: project the candidates tensor in place;
                    elites + mean inherit the projected values, so the CEM mean
                    stays inside the env range. Conceptually clean: the actions
                    the WM scores match what will actually execute.
                  - ``"none"``: disable projection entirely (equivalent to
                    ``env_clip_range=None``).
                Only matters when ``env_clip_range`` is set AND an action
                scaler has been injected; otherwise all three modes degrade to
                no projection.
        """
        self.model = model
        self.batch_size = batch_size
        self.var_scale = var_scale
        self.num_samples = num_samples
        self.n_steps = n_steps
        self.topk = topk
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        # Preserve raw spec; finalised to a broadcast tensor in configure() once
        # physical_dim and action_block are known.
        self.action_clamp = action_clamp
        self._clamp_tensor: torch.Tensor | None = None
        self.binary_gripper = binary_gripper
        # env-space projection buffers — populated by set_action_scaler().
        # Stored as torch tensors on self.device so the projection stays on GPU
        # (no per-iter numpy round trip).
        self.env_clip_range = (
            tuple(env_clip_range) if env_clip_range is not None else None
        )
        if projection_mode not in ("cost_only", "state", "none"):
            raise ValueError(
                f"projection_mode must be 'cost_only' | 'state' | 'none', "
                f"got {projection_mode!r}"
            )
        self.projection_mode = projection_mode
        self._scaler_mean: torch.Tensor | None = None      # (physical_dim,)
        self._scaler_scale: torch.Tensor | None = None     # (physical_dim,)
        self._scaler_mean_tiled: torch.Tensor | None = None    # (action_dim,) cached
        self._scaler_scale_tiled: torch.Tensor | None = None   # (action_dim,) cached

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        """Configure the solver with environment specifications."""
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        if not isinstance(action_space, Box):
            logging.warning(f"Action space is discrete, got {type(action_space)}. CEMSolver may not work as expected.")

        # Finalise action_clamp into a broadcast-ready tensor of shape
        # (action_block * physical_dim,) aligning with candidates' last dim.
        # Scalar stays as a Python float (faster clamp kernel path). A per-dim
        # spec is tiled across action_block so every frame inside a block gets
        # the same per-physical-dim bound.
        self._clamp_tensor = None
        if self.action_clamp is not None and not isinstance(self.action_clamp, (int, float)):
            raw = self.action_clamp
            # OmegaConf ListConfig → list; torch.Tensor → tensor; list/tuple → tensor.
            if hasattr(raw, "_content") and not torch.is_tensor(raw):
                raw = list(raw)
            clamp_t = torch.as_tensor(raw, dtype=torch.float32, device=self.device).flatten()
            if clamp_t.numel() != self._action_dim:
                raise ValueError(
                    f"action_clamp per-dim length {clamp_t.numel()} must equal "
                    f"physical action dim {self._action_dim}"
                )
            if (clamp_t <= 0).any():
                raise ValueError(f"action_clamp entries must be positive, got {clamp_t.tolist()}")
            # Tile across action_block slots: shape (action_block * physical_dim,)
            self._clamp_tensor = clamp_t.repeat(config.action_block).contiguous()
            logging.info(
                f"[CEMSolver] per-dim action_clamp (physical): {clamp_t.tolist()} "
                f"(tiled to action_dim={self._clamp_tensor.numel()})"
            )
        elif isinstance(self.action_clamp, (int, float)):
            logging.info(f"[CEMSolver] scalar action_clamp = {float(self.action_clamp)}")

        # If a scaler was injected before configure(), tile its buffers now
        # that we know action_block.
        self._rebuild_scaler_tiled()

    def set_action_scaler(self, scaler: Any) -> None:
        """Attach an action-space scaler (sklearn StandardScaler or duck-typed).

        When combined with :attr:`env_clip_range`, the CEM loop projects every
        sampled candidate AND the elite mean through env-space: denorm →
        clip(env_clip_range) → renorm. This keeps candidates, elite pool, and
        batch_mean in the env's physical action range, preventing OOD
        exploitation of predictor extrapolation.

        Expected attributes on ``scaler``: ``mean_`` and ``scale_`` each of
        shape ``(physical_dim,)``. sklearn's ``StandardScaler`` satisfies this.

        Passing ``None`` clears the scaler (projection disabled).
        """
        if scaler is None:
            self._scaler_mean = None
            self._scaler_scale = None
            self._scaler_mean_tiled = None
            self._scaler_scale_tiled = None
            logging.info("[CEMSolver] action_scaler cleared; env projection disabled")
            return
        mean_arr = np.asarray(scaler.mean_, dtype=np.float32)
        scale_arr = np.asarray(scaler.scale_, dtype=np.float32)
        self._scaler_mean = torch.from_numpy(mean_arr).to(self.device)
        self._scaler_scale = torch.from_numpy(scale_arr).to(self.device)
        self._rebuild_scaler_tiled()
        if self.env_clip_range is not None:
            logging.info(
                f"[CEMSolver] env projection ENABLED: "
                f"clip={self.env_clip_range}, scaler mean={mean_arr.round(4).tolist()} "
                f"scale={scale_arr.round(4).tolist()}"
            )
        else:
            logging.info(
                "[CEMSolver] action_scaler set but env_clip_range=None — "
                "projection inactive (set env_clip_range to activate)."
            )

    def _rebuild_scaler_tiled(self) -> None:
        """Tile (physical_dim,) scaler buffers to (action_dim,) for broadcast.

        Called by configure() after action_block is known and by
        set_action_scaler() after new stats are injected. No-op when either
        scaler or config is missing.
        """
        if self._scaler_mean is None or not getattr(self, "_configured", False):
            return
        phys = self._scaler_mean.numel()
        if phys != self._action_dim:
            raise ValueError(
                f"action_scaler physical_dim {phys} != action_space physical_dim "
                f"{self._action_dim}; scaler was fit on a different action shape."
            )
        ab = self._config.action_block
        self._scaler_mean_tiled = self._scaler_mean.repeat(ab).contiguous()
        self._scaler_scale_tiled = self._scaler_scale.repeat(ab).contiguous()

    def _project_env(self, x: torch.Tensor) -> torch.Tensor:
        """Env-space projection: denorm → clip(env_clip_range) → renorm.

        Broadcasts against last dim of ``x`` (action_dim). Returns a new
        tensor; callers should reassign. No-op when scaler or env_clip_range
        is missing.
        """
        if (
            self._scaler_mean_tiled is None
            or self.env_clip_range is None
        ):
            return x
        env = x * self._scaler_scale_tiled + self._scaler_mean_tiled
        lo, hi = self.env_clip_range
        env = env.clamp(float(lo), float(hi))
        return (env - self._scaler_mean_tiled) / self._scaler_scale_tiled

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        return self._n_envs

    @property
    def action_dim(self) -> int:
        """Flattened action dimension including action_block grouping."""
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        """Planning horizon in timesteps."""
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        """Make solver callable, forwarding to solve()."""
        return self.solve(*args, **kwargs)

    def init_action_distrib(
        self, actions: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize the action distribution parameters (mean and variance)."""
        var = self.var_scale * torch.ones([self.n_envs, self.horizon, self.action_dim])
        mean = torch.zeros([self.n_envs, 0, self.action_dim]) if actions is None else actions

        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            device = mean.device
            new_mean = torch.zeros([self.n_envs, remaining, self.action_dim])
            mean = torch.cat([mean, new_mean], dim=1).to(device)

        return mean, var

    @torch.inference_mode()
    def solve(
        self, info_dict: dict, init_action: torch.Tensor | None = None
    ) -> dict:
        """Solve the planning problem using Cross Entropy Method."""
        start_time = time.time()
        outputs = {
            "costs": [],
            "mean": [],  # History of means
            "var": [],  # History of vars
        }

        # -- initialize the action distribution globally
        mean, var = self.init_action_distrib(init_action)
        mean = mean.to(self.device)
        var = var.to(self.device)

        total_envs = self.n_envs

        # --- Iterate over batches ---
        for start_idx in range(0, total_envs, self.batch_size):
            end_idx = min(start_idx + self.batch_size, total_envs)
            current_bs = end_idx - start_idx

            # Slice Distribution Parameters for current batch
            batch_mean = mean[start_idx:end_idx]
            batch_var = var[start_idx:end_idx]

            # Expand Info Dict for current batch
            expanded_infos = {}
            for k, v in info_dict.items():
                # v is shape (n_envs, ...)
                # Slice batch
                v_batch = v[start_idx:end_idx]
                if torch.is_tensor(v):
                    # Move to device BEFORE expand, so the expand creates a
                    # zero-copy view on GPU.
                    v_batch = v_batch.to(self.device, non_blocking=True)
                    # Add sample dim: (batch, 1, ...)
                    v_batch = v_batch.unsqueeze(1)
                    # Expand: (batch, num_samples, ...)  — view, stride 0 on sample dim
                    v_batch = v_batch.expand(current_bs, self.num_samples, *v_batch.shape[2:])
                elif isinstance(v, np.ndarray):
                    v_batch = np.repeat(v_batch[:, None, ...], self.num_samples, axis=1)
                expanded_infos[k] = v_batch

            # Optimization Loop
            final_batch_cost = None
            # Per-iter elite-mean cost trace. Collected cheaply (one CPU tolist
            # per iter) and exposed via outputs["iter_cost_history"] so callers
            # can plot convergence or derive a stop heuristic. Not printed by
            # default; see "Enabling per-iter diagnostics" comment below.
            iter_cost_history: list[list[float]] = []

            for step in range(self.n_steps):
                # Sample action sequences: (Batch, Num_Samples, Horizon, Dim)
                candidates = torch.randn(
                    current_bs,
                    self.num_samples,
                    self.horizon,
                    self.action_dim,
                    generator=self.torch_gen,
                    device=self.device,
                )

                # Scale and shift: (Batch, N, H, D) * (Batch, 1, H, D) + (Batch, 1, H, D)
                candidates = candidates * batch_var.unsqueeze(1) + batch_mean.unsqueeze(1)

                # Force the first sample to be the current mean
                candidates[:, 0] = batch_mean

                # Shaping pipeline applied IN ORDER on sampled candidates:
                #   1. norm-space clamp (action_clamp)
                #   2. env-space projection (scaler + env_clip_range)
                #   3. binary gripper (sign snap, last physical dim)
                # Each stage is opt-in via __init__ kwargs; all None/False
                # preserves vanilla CEM behaviour.
                if self._clamp_tensor is not None:
                    candidates = torch.clamp(
                        candidates,
                        min=-self._clamp_tensor,
                        max=self._clamp_tensor,
                    )
                elif isinstance(self.action_clamp, (int, float)):
                    candidates = candidates.clamp(-self.action_clamp, self.action_clamp)

                # Env-space projection — mode-dependent:
                # - "state"     : project candidates IN PLACE. Elites inherit
                #                 projected values → mean bounded to env range.
                # - "cost_only" : project a COPY only for cost eval. Elites
                #                 use pre-projection candidates → mean free
                #                 to drift past env boundary. WM still sees
                #                 in-distribution actions via the copy.
                # - "none"      : no projection.
                if self.projection_mode == "state":
                    candidates = self._project_env(candidates)
                    cost_candidates = candidates
                elif (self.projection_mode == "cost_only"
                        and self._scaler_mean_tiled is not None
                        and self.env_clip_range is not None):
                    cost_candidates = self._project_env(candidates)
                else:
                    cost_candidates = candidates

                # Binarize gripper: snap the LAST physical-dim entry of every
                # action_block slice to sign(x) so gripper is ±1 (hard
                # open/close). Prevents "partial release after grab" where
                # intermediate gripper values cause unreliable grasp
                # maintenance. Applied AFTER clamp so binary values (±1)
                # persist even if clamp < 1 on other dims.
                if self.binary_gripper:
                    physical_dim = self._action_dim
                    action_block = self._config.action_block
                    shape = candidates.shape          # (B, S, H, ab*pd)
                    cand_r = candidates.reshape(*shape[:-1], action_block, physical_dim)
                    grip = cand_r[..., -1]            # (B, S, H, ab)
                    # sign(0) = 0; use >=0 → +1 else -1 to avoid the zero case
                    binarized = torch.where(
                        grip >= 0, torch.ones_like(grip), -torch.ones_like(grip)
                    )
                    cand_r = torch.cat([cand_r[..., :-1], binarized.unsqueeze(-1)], dim=-1)
                    candidates = cand_r.reshape(shape)

                current_info = expanded_infos.copy()

                # Evaluate candidates using the mode-dependent tensor
                # (``cost_candidates`` = projected copy in cost_only mode,
                # ``candidates`` itself in state/none mode — same object when
                # no projection, projected tensor when projection_mode=state).
                costs = self.model.get_cost(current_info, cost_candidates)

                assert isinstance(costs, torch.Tensor), f"Expected cost to be a torch.Tensor, got {type(costs)}"
                assert costs.ndim == 2 and costs.shape[0] == current_bs and costs.shape[1] == self.num_samples, (
                    f"Expected cost to be of shape ({current_bs}, {self.num_samples}), got {costs.shape}"
                )

                # Select Top-K
                # topk_vals: (Batch, K), topk_inds: (Batch, K)
                topk_vals, topk_inds = torch.topk(costs, k=self.topk, dim=1, largest=False)

                # Gather Top-K Candidates
                # We need to select the specific candidates corresponding to topk_inds
                batch_indices = torch.arange(current_bs, device=self.device).unsqueeze(1).expand(-1, self.topk)

                # Indexing: candidates[batch_idx, sample_idx]
                # Result shape: (Batch, K, Horizon, Dim)
                topk_candidates = candidates[batch_indices, topk_inds]

                # Update Mean and Variance based on Top-K
                batch_mean = topk_candidates.mean(dim=1)
                batch_var = topk_candidates.std(dim=1)

                # Clamp mean to prevent cumulative OOD drift across iters.
                # Even if candidates are clamped, mean could still drift if
                # the elite distribution is asymmetric near the boundary.
                # Explicit mean clamp + env projection guarantees bounds.
                if self._clamp_tensor is not None:
                    batch_mean = torch.clamp(
                        batch_mean,
                        min=-self._clamp_tensor,
                        max=self._clamp_tensor,
                    )
                elif isinstance(self.action_clamp, (int, float)):
                    batch_mean = batch_mean.clamp(-self.action_clamp, self.action_clamp)

                # In ``projection_mode="state"`` batch_mean is already bounded
                # because elite candidates were projected before the elite
                # gather. In ``projection_mode="cost_only"`` we INTENTIONALLY
                # skip mean projection — the whole point of the mode is to
                # let the mean drift past the env boundary while only the
                # cost-eval copy is in-distribution. See __init__ docstring.

                # Binarize mean's gripper dim too — elite mean of {-1, +1}
                # samples is in (-1, +1) interior, which defeats binarization.
                # Snap mean back to ±1 so next iter's sampling is centered on
                # a binary gripper.
                if self.binary_gripper:
                    physical_dim = self._action_dim
                    action_block = self._config.action_block
                    shape = batch_mean.shape          # (B, H, ab*pd)
                    mean_r = batch_mean.reshape(*shape[:-1], action_block, physical_dim)
                    grip = mean_r[..., -1]
                    binarized = torch.where(
                        grip >= 0, torch.ones_like(grip), -torch.ones_like(grip)
                    )
                    mean_r = torch.cat([mean_r[..., :-1], binarized.unsqueeze(-1)], dim=-1)
                    batch_mean = mean_r.reshape(shape)

                # Update final cost for logging
                # We average the cost of the top elites
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()
                iter_cost_history.append(final_batch_cost)

            # Write results back to global storage
            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var

            # Store history/metadata
            outputs["costs"].extend(final_batch_cost)
            # Per-batch × per-iter cost trace → appended as one list per env
            # in the batch. Final shape of outputs["iter_cost_history"]:
            # list of ``n_envs`` lists of length ``n_steps``.
            outputs.setdefault("iter_cost_history", []).extend(
                [[row[b] for row in iter_cost_history] for b in range(current_bs)]
            )

        outputs["actions"] = mean.detach().cpu()
        outputs["mean"] = [mean.detach().cpu()]
        outputs["var"] = [var.detach().cpu()]

        print(f"CEM solve time: {time.time() - start_time:.4f} seconds")
        return outputs

    # -----------------------------------------------------------------------
    # Enabling per-iter diagnostics (OPT-IN)
    #
    # The solver stores per-iter elite cost traces in
    # ``outputs["iter_cost_history"]`` but does NOT print them. Add the
    # snippet below at a call site to surface a convergence trace (useful
    # when tuning n_steps / num_samples / var_scale):
    #
    #     out = solver.solve(info_dict)
    #     hist = out["iter_cost_history"]    # list of n_envs × n_steps
    #     for env_i, series in enumerate(hist):
    #         print(f"[env {env_i}] cost {series[0]:.4f} -> {series[-1]:.4f}  "
    #               f"({len(series)} iters)")
    #
    # Or plot via matplotlib / log to wandb. The collection cost is one
    # CPU tolist per CEM iter — negligible compared to the WM forward pass.
    # -----------------------------------------------------------------------
