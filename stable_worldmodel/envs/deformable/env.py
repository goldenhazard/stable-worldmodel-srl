"""Gymnasium adapter for the PyFlex-based deformable env (rope, granular).

Wraps the dino_wm-flavored FlexEnvWrapper (synchronous, batched-by-1 push
actions) in a stable-wm-friendly gymnasium.Env contract: 5-tuple step,
2-tuple reset, pixels returned via render() rather than embedded in obs.

For goal-conditioned planning, reset() samples (init_state, goal_state) via
the underlying FlexEnvWrapper, renders the goal state once into a uint8
HxWx3 image, and returns it as info['goal'] (alongside info['goal_state']
which is the raw particle cloud). step() carries those same fields
forward so stable-wm's MegaWrapper + WorldModelPolicy see a stable goal
image on every tick. chamfer_to_goal() exposes FlexEnvWrapper.eval_state
for the outer MPC loop.

PyFlex itself must be on PYTHONPATH — see dino_wm/install_pyflex.sh. Import
of this module does not import pyflex; that only happens when DeformableEnv
is constructed.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces


_PYFLEX_HINT = (
    "PyFlex import failed. The deformable env needs the compiled PyFlex "
    "bindings on PYTHONPATH. See dino_wm/install_pyflex.sh and the README "
    "in dino_wm/ for the docker-based build; the env vars PYFLEXROOT, "
    "PYTHONPATH, and LD_LIBRARY_PATH must be set (typically via ~/.bashrc)."
)


def _to_uint8_rgb(arr):
    """FlexEnvWrapper renders BGRA-ish (H, W, 5); strip to RGB uint8."""
    arr = np.asarray(arr)
    if arr.shape[-1] >= 3:
        arr = arr[..., :3][..., ::-1]  # BGR -> RGB
    return np.ascontiguousarray(arr.astype(np.uint8))


class DeformableEnv(gym.Env):
    """Gymnasium-wrapped PyFlex deformable env for stable-wm.

    Parameters
    ----------
    object_name : str
        "rope" or "granular". Selects the sim config under
        stable_worldmodel/envs/deformable/conf/env/{object_name}.yaml.
    render_mode : str, optional
        Only "rgb_array" is supported; kept for gymnasium compatibility.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(
        self,
        object_name: str = "rope",
        render_mode: str = "rgb_array",
    ):
        super().__init__()
        try:
            from .FlexEnvWrapper import FlexEnvWrapper  # noqa: F401
        except Exception as e:
            raise ImportError(_PYFLEX_HINT) from e

        self.object_name = object_name
        self.render_mode = render_mode

        from .FlexEnvWrapper import FlexEnvWrapper
        self._sim = FlexEnvWrapper(object_name)

        self.action_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )
        self.observation_space = spaces.Dict(
            {
                "proprio": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "state": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32
                ),
            }
        )

        self._last_pixels = None
        self._last_eef = None
        self._last_state = None
        # Goal-conditioned planning state — populated at reset().
        self._goal_state = None     # (N_particles, 4) torch.Tensor or np.ndarray
        self._goal_image = None     # (H, W, 3) uint8 RGB

    # ------------------------------------------------------------------
    # gymnasium contract
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        sim_seed = seed if seed is not None else 0

        # 1. Resolve goal_state (and a default init_state if none given).
        #    sample_random_init_goal_states returns (init, goal); we keep
        #    goal and let callers override init via options['init_state']
        #    for parity with dino_wm's dataset-init recipe.
        sampled_init, goal_state = self._sim.sample_random_init_goal_states(
            sim_seed
        )
        if "goal_state" in options and options["goal_state"] is not None:
            goal_state = options["goal_state"]
        init_state = options.get("init_state", sampled_init)

        # 2. Render the goal state once (resets sim to goal, grabs frame).
        self._sim.prepare(sim_seed, goal_state)
        goal_image = _to_uint8_rgb(self._sim.get_one_view_img())

        # 3. Reset sim to the init_state we actually want to plan from.
        obs, state_dct = self._sim.prepare(sim_seed, init_state)

        pixels = _to_uint8_rgb(obs["visual"])
        self._last_pixels = pixels
        self._last_eef = state_dct.get("proprio")
        self._last_state = np.asarray(state_dct["state"], dtype=np.float32)
        self._goal_state = goal_state
        self._goal_image = goal_image

        observation = {
            "proprio": np.asarray(obs["proprio"], dtype=np.float32),
            "state": self._last_state.reshape(-1),
        }
        info = self._build_info(pixels)
        return observation, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != 4:
            raise ValueError(
                f"Deformable action must be 4-D (x_s,z_s,x_e,z_e); got {action.shape}"
            )

        obses, rewards, dones, infos = self._sim.step_multiple([action])

        pixels = _to_uint8_rgb(obses["visual"][0])
        proprio = np.asarray(obses["proprio"][0], dtype=np.float32)
        state = np.asarray(infos["state"][0], dtype=np.float32).reshape(-1)

        self._last_pixels = pixels
        self._last_state = state
        self._last_eef = infos.get("pos_agent", [None])[0]

        observation = {"proprio": proprio, "state": state}
        info = self._build_info(pixels)
        # Open-ended pushing — chamfer eval is computed externally by the
        # planner via chamfer_to_goal().
        return observation, 0.0, False, False, info

    def render(self):
        return self._last_pixels

    def close(self):
        # FlexEnv has no explicit teardown; pyflex globals stay alive.
        pass

    # ------------------------------------------------------------------
    # planning helpers used by the outer MPC loop / stable-wm wrappers
    # ------------------------------------------------------------------
    def _build_info(self, pixels):
        """Common info dict — includes goal image+state when available."""
        info = {
            "pixels": pixels,
            "eef_state": self._last_eef,
            "object_name": self.object_name,
        }
        if self._goal_image is not None:
            info["goal"] = self._goal_image
        if self._goal_state is not None:
            # numpy-ify so vectorized wrappers can stack across envs
            gs = self._goal_state
            if hasattr(gs, "numpy"):
                gs = gs.numpy()
            info["goal_state"] = np.asarray(gs, dtype=np.float32)
        return info

    def chamfer_to_goal(self):
        """Return FlexEnvWrapper.eval_state(goal_state, current_state).

        Mirrors dino_wm exactly — success threshold is the same buggy
        `CD < 0` check, so success is always False; chamfer_distance is
        the load-bearing number.

        FlexEnvWrapper.eval_state does ``torch.tensor([goal_state])`` which
        only works when both inputs are numpy arrays (dino_wm's calling
        convention) — convert defensively.
        """
        if self._goal_state is None:
            raise RuntimeError(
                "chamfer_to_goal() called before reset() — no goal_state cached"
            )
        gs = self._goal_state
        if hasattr(gs, "numpy"):
            gs = gs.numpy()
        gs = np.asarray(gs, dtype=np.float32)
        cur_state = np.asarray(self._sim.get_states(), dtype=np.float32)
        return self._sim.eval_state(gs, cur_state)

    def sample_random_init_goal_states(self, seed):
        """Proxy to FlexEnvWrapper for callers that want to pre-sample states
        before constructing per-env reset options."""
        return self._sim.sample_random_init_goal_states(seed)
