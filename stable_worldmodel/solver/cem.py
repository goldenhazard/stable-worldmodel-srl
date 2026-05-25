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
        # === MPCC additions (all default None preserves MPC behavior) ===
        physical_dim: int | None = None,
        nu_clamp: tuple | list | None = None,
        nu_init: float | None = None,
        nu_var_scale: float | None = None,
        phys_per_step: int | None = None,
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
            physical_dim: MPCC support. Number of leading slots in the flat
                action_dim that are physical (rest are virtual velocity ν).
                ``None`` (default) → all action_dim is physical (MPC behavior).
                For cube AW MPCC: ``physical_dim=25`` (= 5 phys × action_block 5),
                with one ν slot, total ``action_dim=26``.
            nu_clamp: MPCC support. ``[lo, hi]`` clamp applied to ν slot at
                every iter. Idempotent — safe to apply each iter without
                cumulative drift. ``None`` → no ν clamp.
            nu_init: MPCC support. Initial batch_mean for ν slot in cold-start
                horizon timesteps. ``None`` → ν starts at 0.
            nu_var_scale: MPCC support. Initial batch_var for ν slot.
                ``None`` → uses ``var_scale`` for ν (same as physical).
            phys_per_step: Env action dim per env-step — required to locate the
                gripper dim within a flat phys block of shape
                ``(outer_action_block × phys_per_step)``. ``None`` (default)
                falls back to ``_action_dim`` (correct for MPC, where the
                action_space passed to CEM IS the env action space). MPCC must
                set this explicitly (the inner CEM sees an augmented
                action_dim that includes ν, so ``_action_dim`` is wrong);
                ``latent_mpcc.MPCCSolver.configure`` injects the correct value
                programmatically.
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
        # MPCC slot config (finalised in configure()).
        self.physical_dim = physical_dim
        self.nu_clamp = tuple(nu_clamp) if nu_clamp is not None else None
        self.nu_init = nu_init
        self.nu_var_scale = nu_var_scale
        self._phys_slice: slice = slice(None)
        self._nu_slice: slice | None = None
        # Gripper-layout helper. ``phys_per_step`` is the env action dim per
        # env-step (5 for cube, 2 for PushT). ``_phys_per_step`` is finalised
        # in configure() — defaults to _action_dim for MPC, must be set
        # explicitly for MPCC.
        self.phys_per_step = phys_per_step
        self._phys_per_step: int = 0   # finalised in configure()

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        """Configure the solver with environment specifications."""
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        self._configured = True

        if not isinstance(action_space, Box):
            logging.warning(f"Action space is discrete, got {type(action_space)}. CEMSolver may not work as expected.")

        # MPCC slot setup. ``physical_dim`` is the FLAT count of leading phys
        # slots (= phys_per_step × outer_action_block, e.g. 25 for cube AW).
        # ``None`` → all action_dim is physical (MPC behavior preserved).
        flat_action_dim = self._action_dim * config.action_block
        if self.physical_dim is None or self.physical_dim >= flat_action_dim:
            self._phys_slice = slice(None)
            self._nu_slice = None
            flat_phys_dim = flat_action_dim
        else:
            self._phys_slice = slice(0, self.physical_dim)
            self._nu_slice = slice(self.physical_dim, None)
            flat_phys_dim = self.physical_dim
            logging.info(
                f"[CEMSolver] MPCC mode: physical_dim={self.physical_dim} of "
                f"flat_action_dim={flat_action_dim}; ν slot size="
                f"{flat_action_dim - self.physical_dim}"
            )

        # Finalise phys_per_step (gripper-dim layout helper). MPC default =
        # _action_dim (action_space passed to CEM IS env action space).
        # MPCC must inject the env action dim explicitly because inner CEM
        # sees an augmented action_dim that includes ν.
        self._phys_per_step = (
            self._action_dim if self.phys_per_step is None else int(self.phys_per_step)
        )
        if flat_phys_dim % self._phys_per_step != 0:
            raise ValueError(
                f"phys_per_step={self._phys_per_step} must evenly divide flat "
                f"phys dim {flat_phys_dim}"
            )

        # Finalise action_clamp into a broadcast-ready tensor of shape
        # ``(flat_phys_dim,)`` aligning with the phys slice of candidates.
        # Scalar stays as a Python float. Per-dim spec is tiled by
        # ``flat_phys_dim / clamp_t.numel()`` (= action_block for MPC, =
        # outer-action_block for MPCC).
        self._clamp_tensor = None
        if self.action_clamp is not None and not isinstance(self.action_clamp, (int, float)):
            raw = self.action_clamp
            # OmegaConf ListConfig → list; torch.Tensor → tensor; list/tuple → tensor.
            if hasattr(raw, "_content") and not torch.is_tensor(raw):
                raw = list(raw)
            clamp_t = torch.as_tensor(raw, dtype=torch.float32, device=self.device).flatten()
            if flat_phys_dim % clamp_t.numel() != 0:
                raise ValueError(
                    f"action_clamp dim {clamp_t.numel()} must evenly divide "
                    f"flat phys dim {flat_phys_dim}"
                )
            if (clamp_t <= 0).any():
                raise ValueError(f"action_clamp entries must be positive, got {clamp_t.tolist()}")
            tile = flat_phys_dim // clamp_t.numel()
            self._clamp_tensor = clamp_t.repeat(tile).contiguous()
            logging.info(
                f"[CEMSolver] per-dim action_clamp (physical): {clamp_t.tolist()} "
                f"(tiled ×{tile} → phys_dim={self._clamp_tensor.numel()})"
            )
        elif isinstance(self.action_clamp, (int, float)):
            logging.info(f"[CEMSolver] scalar action_clamp = {float(self.action_clamp)}")

        # If a scaler was injected before configure(), tile its buffers now.
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
        """Tile (per-env-phys,) scaler buffers to (flat_phys_dim,) for broadcast.

        Called by configure() after action_block is known and by
        set_action_scaler() after new stats are injected. No-op when either
        scaler or config is missing.

        Tile factor = flat_phys_dim / scaler.numel(). This is action_block for
        MPC (per-env-phys × action_block = action_dim) and outer-action_block
        for MPCC (per-env-phys × outer-action_block = physical_dim).
        """
        if self._scaler_mean is None or not getattr(self, "_configured", False):
            return
        scaler_dim = self._scaler_mean.numel()
        # Determine target flat phys dim (MPC: full action_dim*ab; MPCC: physical_dim).
        if self.physical_dim is not None:
            flat_phys_dim = self.physical_dim
        else:
            flat_phys_dim = self._action_dim * self._config.action_block
        if flat_phys_dim % scaler_dim != 0:
            raise ValueError(
                f"action_scaler dim {scaler_dim} must evenly divide flat phys "
                f"dim {flat_phys_dim}; scaler was fit on a different action shape."
            )
        tile = flat_phys_dim // scaler_dim
        self._scaler_mean_tiled = self._scaler_mean.repeat(tile).contiguous()
        self._scaler_scale_tiled = self._scaler_scale.repeat(tile).contiguous()

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
        """Initialize the action distribution parameters (mean and variance).

        MPCC: ν slot gets ``nu_init`` mean and ``nu_var_scale`` var when set
        (cold-start slots only). Phys slot uses default zeros mean and
        ``var_scale`` var. MPC path (``_nu_slice is None``) is unchanged.
        """
        var = self.var_scale * torch.ones([self.n_envs, self.horizon, self.action_dim])
        if self._nu_slice is not None and self.nu_var_scale is not None:
            var[..., self._nu_slice] = self.nu_var_scale

        mean = torch.zeros([self.n_envs, 0, self.action_dim]) if actions is None else actions

        remaining = self.horizon - mean.shape[1]
        if remaining > 0:
            device = mean.device
            new_mean = torch.zeros([self.n_envs, remaining, self.action_dim])
            if self._nu_slice is not None and self.nu_init is not None:
                new_mean[..., self._nu_slice] = self.nu_init
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

            # Init / final elite-mean capture for the
            # ``[env i] cost A / l2 X → B / l2 Y`` log line.
            # ``cost`` = weighted total (already in topk_vals).
            # ``l2``   = raw contour / tracking term, fetched from
            # ``cost_model._last_breakdown[{tracking_raw|contour_raw}]``
            # (MPC: tracking_raw = Σ_t d(z_pred, z_ref); MPCC: contour_raw =
            # Σ_t ‖z_pred − z_ref‖²). Either key — whichever is present.
            init_total_per_env: torch.Tensor | None = None
            init_l2_per_env: torch.Tensor | None = None
            final_l2_per_env: torch.Tensor | None = None
            # Phys / ν elite-mean stats for the log line. ``|a|`` = mean abs of
            # the phys slice over (H, A_phys) per env; ``ν`` = mean of the ν
            # slice over H per env. ν is None for MPC (``_nu_slice is None``).
            init_aabs_per_env: torch.Tensor | None = None
            final_aabs_per_env: torch.Tensor | None = None
            init_nu_per_env: torch.Tensor | None = None
            final_nu_per_env: torch.Tensor | None = None

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
                #   1. norm-space clamp (action_clamp)        → phys slot
                #   2. env-space projection (scaler + clip)   → phys slot
                #   3. binary gripper (sign snap, last phys)  → phys slot (MPC only)
                #   4. ν clamp (nu_clamp)                      → ν slot (MPCC only)
                # MPC path (``_nu_slice is None``): all shaping operates on the
                # full ``candidates`` (phys_slice = slice(None) is a no-op view).
                # MPCC path (``_nu_slice is not None``): shaping on phys slice
                # only via ``candidates[..., _phys_slice]``; ν is clamped
                # separately. Single ``candidates`` tensor preserves
                # cost==elite invariant.
                if self._nu_slice is None:
                    # === MPC path (verbatim original behavior) ===
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
                else:
                    # === MPCC path: shape phys slice only, ν via nu_clamp ===
                    phys = candidates[..., self._phys_slice]
                    # 1. norm-space clamp on phys
                    if self._clamp_tensor is not None:
                        phys = torch.clamp(
                            phys,
                            min=-self._clamp_tensor,
                            max=self._clamp_tensor,
                        )
                    elif isinstance(self.action_clamp, (int, float)):
                        phys = phys.clamp(-self.action_clamp, self.action_clamp)
                    # 2. env-space projection on phys (state/cost_only treated
                    #    the same for MPCC — always in-place; cost_only mode
                    #    not supported in MPCC. "none" skips projection.)
                    if (self.projection_mode != "none"
                            and self._scaler_mean_tiled is not None
                            and self.env_clip_range is not None):
                        phys = self._project_env(phys)
                    # 3. binary_gripper: skipped for MPCC (would conflict with
                    #    ν semantics; MPCC yaml should set binary_gripper=False).
                    candidates[..., self._phys_slice] = phys

                    # 4. ν clamp (idempotent, safe to apply each iter)
                    if self.nu_clamp is not None:
                        nu = candidates[..., self._nu_slice]
                        nu = nu.clamp(*self.nu_clamp)
                        candidates[..., self._nu_slice] = nu

                    # cost == elite: single candidates tensor.
                    cost_candidates = candidates

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
                if self._nu_slice is None:
                    # MPC path: full batch_mean clamp (verbatim original)
                    if self._clamp_tensor is not None:
                        batch_mean = torch.clamp(
                            batch_mean,
                            min=-self._clamp_tensor,
                            max=self._clamp_tensor,
                        )
                    elif isinstance(self.action_clamp, (int, float)):
                        batch_mean = batch_mean.clamp(-self.action_clamp, self.action_clamp)

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
                else:
                    # MPCC path: clamp phys mean and ν mean separately.
                    if self._clamp_tensor is not None:
                        phys_mean = batch_mean[..., self._phys_slice]
                        phys_mean = torch.clamp(
                            phys_mean,
                            min=-self._clamp_tensor,
                            max=self._clamp_tensor,
                        )
                        batch_mean[..., self._phys_slice] = phys_mean
                    elif isinstance(self.action_clamp, (int, float)):
                        phys_mean = batch_mean[..., self._phys_slice]
                        phys_mean = phys_mean.clamp(-self.action_clamp, self.action_clamp)
                        batch_mean[..., self._phys_slice] = phys_mean
                    # ν mean clamp: keeps next-iter sampling within nu_clamp.
                    if self.nu_clamp is not None:
                        nu_mean = batch_mean[..., self._nu_slice]
                        nu_mean = nu_mean.clamp(*self.nu_clamp)
                        batch_mean[..., self._nu_slice] = nu_mean

                # Update final cost for logging
                # We average the cost of the top elites
                final_batch_cost = topk_vals.mean(dim=1).cpu().tolist()
                iter_cost_history.append(final_batch_cost)

                # Capture init / final elite-mean L2 (raw contour term) for
                # the per-env log line. weighted total at first iter caches
                # via ``init_total_per_env``; at last iter, ``final_batch_cost``
                # itself is the weighted total. ``getattr(self.model,
                # "_last_breakdown", {})`` is empty for cost models that
                # don't populate it → l2 stays None and the log line shows nan.
                if step == 0 or step == self.n_steps - 1:
                    bd = getattr(self.model, "_last_breakdown", {}) or {}
                    raw_l2 = bd.get("tracking_raw")
                    if raw_l2 is None:
                        raw_l2 = bd.get("contour_raw")
                    # Phys / ν elite-mean stats from batch_mean (= updated
                    # topk_candidates.mean above). For MPCC, _phys_slice
                    # selects phys; _nu_slice selects ν. For MPC, _phys_slice
                    # = slice(None) and _nu_slice = None.
                    phys_mean_now = batch_mean[..., self._phys_slice]
                    aabs_now = phys_mean_now.abs().mean(dim=(-1, -2)).detach().cpu()
                    nu_now = None
                    if self._nu_slice is not None:
                        nu_mean_now = batch_mean[..., self._nu_slice]
                        nu_now = nu_mean_now.mean(dim=(-1, -2)).detach().cpu()
                    if step == 0:
                        init_total_per_env = topk_vals.mean(dim=1).detach().cpu()
                        init_aabs_per_env = aabs_now
                        init_nu_per_env = nu_now
                        if raw_l2 is not None:
                            init_l2_per_env = raw_l2.gather(1, topk_inds).mean(dim=1).detach().cpu()
                    if step == self.n_steps - 1:
                        final_aabs_per_env = aabs_now
                        final_nu_per_env = nu_now
                        if raw_l2 is not None:
                            final_l2_per_env = raw_l2.gather(1, topk_inds).mean(dim=1).detach().cpu()

            # Write results back to global storage
            mean[start_idx:end_idx] = batch_mean
            var[start_idx:end_idx] = batch_var

            # Per-env init→final convergence line (one row per env in this batch).
            # weighted total = ``cost``; raw L2 (tracking-only) = ``l2``;
            # ν = elite-mean ν (MPCC only); |a| = elite-mean |phys|.
            for _b in range(current_bs):
                _env_g = start_idx + _b
                _it = (init_total_per_env[_b].item()
                       if init_total_per_env is not None else float("nan"))
                _il = (init_l2_per_env[_b].item()
                       if init_l2_per_env is not None else float("nan"))
                _ia = (init_aabs_per_env[_b].item()
                       if init_aabs_per_env is not None else float("nan"))
                _in = (init_nu_per_env[_b].item()
                       if init_nu_per_env is not None else None)
                _ft = final_batch_cost[_b]
                _fl = (final_l2_per_env[_b].item()
                       if final_l2_per_env is not None else float("nan"))
                _fa = (final_aabs_per_env[_b].item()
                       if final_aabs_per_env is not None else float("nan"))
                _fn = (final_nu_per_env[_b].item()
                       if final_nu_per_env is not None else None)
                if _in is not None and _fn is not None:
                    print(
                        f"  [env {_env_g}] cost {_it:.4f} / l2 {_il:.4f} / "
                        f"ν {_in:.3f} / |a| {_ia:.3f}  →  "
                        f"{_ft:.4f} / l2 {_fl:.4f} / ν {_fn:.3f} / |a| {_fa:.3f}  "
                        f"({self.n_steps} iters)"
                    )
                else:
                    print(
                        f"  [env {_env_g}] cost {_it:.4f} / l2 {_il:.4f} / "
                        f"|a| {_ia:.3f}  →  "
                        f"{_ft:.4f} / l2 {_fl:.4f} / |a| {_fa:.3f}  "
                        f"({self.n_steps} iters)"
                    )

            # Store history/metadata
            outputs["costs"].extend(final_batch_cost)
            # Per-batch × per-iter cost trace → appended as one list per env
            # in the batch. Final shape of outputs["iter_cost_history"]:
            # list of ``n_envs`` lists of length ``n_steps``.
            outputs.setdefault("iter_cost_history", []).extend(
                [[row[b] for row in iter_cost_history] for b in range(current_bs)]
            )

            # Per-env init/final raw L2 + |a| + ν elite-mean stats. None →
            # NaN so the per-env list shape is uniform across batches.
            def _to_list(t, n):
                return t.tolist() if t is not None else [float("nan")] * n
            outputs.setdefault("init_l2",   []).extend(_to_list(init_l2_per_env,   current_bs))
            outputs.setdefault("final_l2",  []).extend(_to_list(final_l2_per_env,  current_bs))
            outputs.setdefault("init_aabs", []).extend(_to_list(init_aabs_per_env, current_bs))
            outputs.setdefault("final_aabs",[]).extend(_to_list(final_aabs_per_env,current_bs))
            outputs.setdefault("init_nu",   []).extend(_to_list(init_nu_per_env,   current_bs))
            outputs.setdefault("final_nu",  []).extend(_to_list(final_nu_per_env,  current_bs))

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
