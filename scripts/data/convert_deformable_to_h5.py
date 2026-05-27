"""Convert dino_wm-format deformable datasets (rope, granular) to stable-wm HDF5.

Source layout (one folder per object under SRC_ROOT):
    SRC_ROOT/{object_name}/
        states.pth          (N_rollout, T, P, 4)   particle positions+mass
        actions.pth         (N_rollout, T, 4)      4D push (x_s, z_s, x_e, z_e)
        {ep:06d}/obses.pth  (T, H, W, 3) uint8     per-episode rendered frames

Target layout (one .h5 per split under $STABLEWM_HOME/datasets/):
    deformable_{obj}_expert_train.h5  /  deformable_{obj}_expert_val.h5
        pixels      (N_steps, H, W, 3) uint8
        action      (N_steps, 4)       float32
        proprio     (N_steps, 1)       float32     dummy zeros (dino_wm convention)
        state       (N_steps, P*4)     float32     flattened particle positions
        ep_offset   (N_eps,)           int64       start index of each episode
        ep_len      (N_eps,)           int32       length of each episode

Usage:
    python scripts/data/convert_deformable_to_h5.py \\
        --src /mnt/HDD1/proj_joint_generation_ext/dino_wm/data/deformable \\
        --objects rope granular \\
        --split-ratio 0.9

Set STABLEWM_HOME if you want the .h5 files written somewhere other than
~/.stable_worldmodel/. The script writes uncompressed h5 (debug-friendly);
compression can be added later if disk pressure shows up.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

# Import stable_worldmodel only to resolve the cache dir consistently.
from stable_worldmodel.data.utils import get_cache_dir


def convert_one(
    src_root: Path,
    object_name: str,
    out_dir: Path,
    split_ratio: float,
    n_rollout_cap: int | None,
) -> None:
    """Convert one object (rope|granular) into train + val .h5 files."""
    obj_dir = src_root / object_name
    if not obj_dir.is_dir():
        raise FileNotFoundError(f"No source folder at {obj_dir}")

    print(f"\n=== {object_name} ===")
    print(f"  src: {obj_dir}")
    states = torch.load(obj_dir / "states.pth", map_location="cpu").float()
    actions = torch.load(obj_dir / "actions.pth", map_location="cpu").float()
    # states: (N, T, P, 4)  →  flatten last two dims to (N, T, P*4)
    n_total, T, P, D = states.shape
    states = states.reshape(n_total, T, P * D)
    assert actions.shape[:2] == (n_total, T), (
        f"action/state mismatch: {actions.shape} vs {states.shape}"
    )

    if n_rollout_cap is not None:
        n_total = min(n_total, n_rollout_cap)
        states = states[:n_total]
        actions = actions[:n_total]

    print(f"  episodes={n_total} timesteps={T} particles={P} action_dim={actions.shape[-1]}")

    # Train/val episode split is contiguous (matches stable-wm's pusht_expert_*
    # convention of one file per split, no within-file split logic).
    n_train = int(round(n_total * split_ratio))
    split_idx = {"train": (0, n_train), "val": (n_train, n_total)}

    for split_name, (lo, hi) in split_idx.items():
        if hi <= lo:
            print(f"  [{split_name}] no episodes — skipping")
            continue

        n_eps = hi - lo
        n_steps = n_eps * T
        out_path = out_dir / f"deformable_{object_name}_expert_{split_name}.h5"
        print(f"  [{split_name}] → {out_path}  ({n_eps} eps, {n_steps} steps)")

        # We need to know image shape to size the pixels dataset; peek at first ep.
        first_ep_imgs = torch.load(
            obj_dir / f"{lo:06d}" / "obses.pth", map_location="cpu"
        )
        if first_ep_imgs.shape[0] != T:
            raise RuntimeError(
                f"Episode {lo:06d} has {first_ep_imgs.shape[0]} frames, "
                f"expected {T} to match states/actions."
            )
        H, W, C = first_ep_imgs.shape[1:]
        action_dim = actions.shape[-1]
        state_dim = states.shape[-1]

        with h5py.File(out_path, "w") as f:
            pixels_d = f.create_dataset(
                "pixels", shape=(n_steps, H, W, C), dtype=np.uint8,
                chunks=(min(T, 64), H, W, C),
            )
            action_d = f.create_dataset(
                "action", shape=(n_steps, action_dim), dtype=np.float32,
            )
            proprio_d = f.create_dataset(
                "proprio", shape=(n_steps, 1), dtype=np.float32,
            )
            state_d = f.create_dataset(
                "state", shape=(n_steps, state_dim), dtype=np.float32,
            )
            ep_offset_d = f.create_dataset(
                "ep_offset", shape=(n_eps,), dtype=np.int64,
            )
            ep_len_d = f.create_dataset(
                "ep_len", shape=(n_eps,), dtype=np.int32,
            )

            # First episode is already loaded; reuse it.
            write_cursor = 0
            for local_i, global_ep in enumerate(tqdm(range(lo, hi), desc=f"{object_name}/{split_name}")):
                if local_i == 0:
                    imgs = first_ep_imgs
                else:
                    imgs = torch.load(
                        obj_dir / f"{global_ep:06d}" / "obses.pth",
                        map_location="cpu",
                    )
                if imgs.shape != (T, H, W, C):
                    raise RuntimeError(
                        f"Episode {global_ep:06d}: image shape {imgs.shape} "
                        f"differs from first episode {(T, H, W, C)}."
                    )

                start, end = write_cursor, write_cursor + T
                pixels_d[start:end] = imgs.numpy().astype(np.uint8)
                action_d[start:end] = actions[global_ep].numpy().astype(np.float32)
                proprio_d[start:end] = 0.0
                state_d[start:end] = states[global_ep].numpy().astype(np.float32)
                ep_offset_d[local_i] = start
                ep_len_d[local_i] = T
                write_cursor = end

        print(f"  [{split_name}] wrote {out_path.stat().st_size / 1e9:.2f} GB")


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("/mnt/HDD1/proj_joint_generation_ext/dino_wm/data/deformable"),
        help="Source root holding {object_name}/ subfolders.",
    )
    parser.add_argument(
        "--objects",
        nargs="+",
        default=["rope", "granular"],
        choices=["rope", "granular"],
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=1.0,
        help=(
            "Fraction of episodes written to the train file (rest go to a "
            "val file). Default 1.0 = single train file; both prejepa.py "
            "and lewm.py do their own 0.9/0.1 split on top via train_split."
        ),
    )
    parser.add_argument(
        "--n-rollout-cap",
        type=int,
        default=None,
        help="If set, limit to this many episodes per object (for debugging).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Destination dir; defaults to $STABLEWM_HOME/datasets/.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or get_cache_dir(sub_folder="datasets")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    for obj in args.objects:
        convert_one(
            src_root=args.src,
            object_name=obj,
            out_dir=out_dir,
            split_ratio=args.split_ratio,
            n_rollout_cap=args.n_rollout_cap,
        )


if __name__ == "__main__":
    main()
