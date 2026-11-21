"""Select and apply new-position anchor merges for HAC++ scene training."""

from collections import Counter
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from torch.nn import functional as F


ANCHOR_FIELDS = ("_anchor", "_offset", "_mask", "_anchor_feat", "_scaling",
                 "_rotation", "_opacity")
EXPECTED_SAVING_BYTES = 1.0
RESPONSE_RMS_LIMIT = 0.05


def predict_rgba(state, xyz, features, cameras):
    """Premultiplied RGB and opacity before rasterization, in slot order."""
    vectors = torch.from_numpy(xyz[:, None] - cameras[None]).float()
    distances = vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    features = torch.from_numpy(features).float()
    inputs = torch.cat((features[:, None].expand(-1, len(cameras), -1),
                        vectors / distances, distances), dim=-1)

    def predict(name):
        first = F.linear(inputs, state[f"{name}.0.weight"], state[f"{name}.0.bias"])
        return F.linear(first.relu(), state[f"{name}.2.weight"], state[f"{name}.2.bias"])

    shape = (len(xyz), len(cameras), -1, 1)
    opacity = predict("mlp_opacity").tanh().clamp_min(0).reshape(shape)
    rgb = predict("mlp_color").sigmoid().reshape((*shape[:-1], 3))
    rgba = torch.cat((rgb * opacity, opacity), dim=-1)
    return rgba.transpose(1, 2).reshape(len(xyz), opacity.shape[2], -1).numpy()


def nearest_free_grid(ideal, grid_step, occupied, low, high):
    base = np.clip(np.rint(ideal / grid_step).astype(np.int64), low, high)
    base_distance = (base * grid_step - ideal) ** 2
    other_axes = base_distance.sum() - base_distance
    radius = 0
    while True:
        lower = np.maximum(base - radius, low)
        upper = np.minimum(base + radius, high)
        axes = [np.arange(lo, hi + 1) for lo, hi in zip(lower, upper)]
        sites = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        distances = ((sites * grid_step - ideal) ** 2).sum(axis=1)
        for index in np.argsort(distances, kind="stable"):
            site = sites[index]
            if tuple(site) in occupied:
                continue
            # The nearest unsearched site must cross a face of the cube.
            # Use the closest legal coordinates on the other two axes, even
            # when the continuous centroid is outside the scene bounds.
            below_distance = ((lower - 1) * grid_step - ideal) ** 2 + other_axes
            above_distance = ((upper + 1) * grid_step - ideal) ** 2 + other_axes
            outside = min(below_distance[lower > low].min(initial=np.inf),
                          above_distance[upper < high].min(initial=np.inf))
            if distances[index] <= outside:
                return site
            break
        if np.array_equal(lower, low) and np.array_equal(upper, high):
            return None
        radius += 1


@torch.no_grad()
def plan_anchor_merge(state, grid_step, meta, camera_centers, *,
                      response_rms_limit=RESPONSE_RMS_LIMIT,
                      min_saving_bytes=EXPECTED_SAVING_BYTES, candidate_neighbors=None):
    if min_saving_bytes is None and candidate_neighbors is None:
        candidate_neighbors = 16
    if candidate_neighbors is not None and candidate_neighbors < 1:
        raise ValueError("candidate_neighbors must be positive")
    arrays = {key: state[key].numpy().copy() for key in ANCHOR_FIELDS}
    anchor = arrays["_anchor"]
    grid = np.rint(anchor / grid_step).astype(np.int64)
    centers = grid.astype(np.float64) * grid_step
    low, high = grid.min(axis=0), grid.max(axis=0)
    occupied = Counter(map(tuple, grid))
    offset = arrays["_offset"]
    capacity = offset.shape[1]
    mask = arrays["_mask"]
    active = (state["_mask"][:, :capacity, 0].sigmoid() > 0.01).numpy()
    counts = active.sum(axis=1)
    features = arrays["_anchor_feat"]
    scaling = arrays["_scaling"]
    position_scale = np.exp(scaling[:, :3])
    initial_anchors = int(np.count_nonzero(counts))
    initial_slots = int(counts.sum())
    if initial_anchors != meta["num_anchors"]:
        raise ValueError("Checkpoint and reference bitstream have different active anchor counts")
    empty_report = {
        "initial_coded_anchors": initial_anchors,
        "final_coded_anchors": initial_anchors,
        "merges": 0,
        "active_offsets": initial_slots,
        "response_rms_max": response_rms_limit,
        "expected_saving_min_bytes": min_saving_bytes,
        "candidate_neighbors": candidate_neighbors,
        "rounds": [],
    }
    if initial_anchors < 2 or not meta["bit_offsets"]:
        return arrays, np.arange(len(anchor)), np.empty(0, dtype=np.int64), empty_report
    shared_bytes = sum(meta[key] for key in
                       ("bit_anchor", "bit_feat", "bit_scaling", "bit_masks")) / (8 * initial_anchors)
    offset_bytes = meta["bit_offsets"] / (8 * initial_slots)
    if min_saving_bytes is not None and shared_bytes <= min_saving_bytes:
        return arrays, np.arange(len(anchor)), np.empty(0, dtype=np.int64), empty_report
    live = np.ones(len(anchor), dtype=bool)
    merged = np.zeros(len(anchor), dtype=bool)
    spacing, _ = cKDTree(centers[counts > 0]).query(centers[counts > 0], k=2)
    typical_spacing = max(grid_step, float(np.median(spacing[:, 1])))
    # The displacement term alone makes larger pairs fail the byte test.
    # This tree radius is its derived search bound, not an extra distance rule.
    if candidate_neighbors is None:
        search_bound = typical_spacing * np.sqrt((shared_bytes - min_saving_bytes)
                                                  / offset_bytes)
    response = predict_rgba(state, centers, features, camera_centers)
    history = []
    max_position_error = 0.0
    rms_response = []
    savings = []
    while True:
        usable = np.flatnonzero(live & (counts > 0) & (counts < capacity))
        if len(usable) < 2:
            break
        tree = cKDTree(centers[usable])
        if candidate_neighbors is None:
            pairs = usable[tree.query_pairs(search_bound, output_type="ndarray")]
        else:
            _, nearest = tree.query(centers[usable], k=min(len(usable), candidate_neighbors + 1))
            local_rows = np.broadcast_to(np.arange(len(usable))[:, None], nearest.shape)
            distinct = local_rows != nearest
            local_pairs = np.stack((local_rows[distinct], nearest[distinct]), axis=1)
            local_pairs.sort(axis=1)
            pairs = usable[np.unique(local_pairs, axis=0)]
        if len(pairs):
            pairs = pairs[counts[pairs].sum(axis=1) <= capacity]
        separation = np.linalg.norm(centers[pairs[:, 0]] - centers[pairs[:, 1]], axis=1)
        available = shared_bytes - offset_bytes * (separation / typical_spacing) ** 2
        order = np.lexsort((pairs[:, 1], pairs[:, 0], -available))
        used = set()
        changed = []
        rejected = {"conflict": 0, "grid": 0, "bytes": 0, "response": 0}
        for index in order:
            left, right = map(int, pairs[index])
            if left in used or right in used:
                rejected["conflict"] += 1
                continue
            left_slots = np.flatnonzero(active[left])
            right_slots = np.flatnonzero(active[right])
            owners = np.repeat([left, right], [len(left_slots), len(right_slots)])
            old_slots = np.concatenate((left_slots, right_slots))
            old_offset = offset[owners, old_slots]
            world = centers[owners] + position_scale[owners] * old_offset
            site = nearest_free_grid(world.mean(axis=0), grid_step, occupied, low, high)
            if site is None:
                rejected["grid"] += 1
                continue
            center = site.astype(np.float64) * grid_step
            log_scale = np.average(scaling[owners], axis=0)
            new_position_scale = np.exp(log_scale[:3])
            new_offsets = ((world - center) / new_position_scale).astype(offset.dtype)
            extra_symbols = np.log2(1 + np.abs(new_offsets) / 0.2).sum()
            extra_symbols -= np.log2(1 + np.abs(old_offset) / 0.2).sum()
            saving = available[index] - max(0, extra_symbols / 8)
            if min_saving_bytes is not None and saving < min_saving_bytes:
                rejected["bytes"] += 1
                continue
            feat = np.average(features[owners], axis=0).astype(features.dtype)
            candidate = predict_rgba(state, center[None], feat[None], camera_centers)[0]
            delta = response[owners, old_slots, None] - candidate[None]
            errors = np.mean(delta ** 2, axis=-1)
            if float(errors.min(axis=1).mean()) > response_rms_limit ** 2:
                rejected["response"] += 1
                continue
            source_rows, new_slots = linear_sum_assignment(errors)
            error = float(np.sqrt(errors[source_rows, new_slots].mean()))
            if error > response_rms_limit:
                rejected["response"] += 1
                continue

            old_mask_values = mask[owners, old_slots, 0].copy()
            mask[left, :capacity] = -10.0
            mask[left, new_slots, 0] = old_mask_values[source_rows]
            active[left] = False
            active[left, new_slots] = True
            offset[left] = 0
            offset[left, new_slots] = new_offsets[source_rows]
            counts[left] = len(source_rows)
            features[left] = feat
            scaling[left] = log_scale
            position_scale[left] = new_position_scale
            for original in (tuple(grid[left]), tuple(grid[right])):
                occupied[original] -= 1
                if occupied[original] == 0:
                    del occupied[original]
            occupied[tuple(site)] = 1
            grid[left] = site
            centers[left] = center
            anchor[left] = center
            response[left] = candidate
            live[right] = False
            merged[left] = True
            used.update((left, right))
            changed.append((left, right))
            recovered = anchor[left] + position_scale[left] * offset[left, new_slots]
            max_position_error = max(max_position_error, float(np.abs(
                recovered - world[source_rows]).max()))
            rms_response.append(error)
            savings.append(float(saving))

        history.append({"candidates": len(pairs), "accepted": len(changed),
                        "rejected": rejected})
        print(f"[Merge] round={len(history)} candidates={len(pairs)} "
              f"accepted={len(changed)}", flush=True)
        if not changed:
            break

    rows = np.flatnonzero(live)
    remaining_slots = int(active[rows].sum())
    merges = int(len(anchor) - len(rows))
    if remaining_slots != initial_slots or sum(occupied.values()) != len(rows):
        raise RuntimeError(f"Invalid merged checkpoint: merges={merges}, "
                           f"slots={remaining_slots}/{initial_slots}, sites={len(occupied)}")
    report = {
        "initial_coded_anchors": initial_anchors,
        "final_coded_anchors": initial_anchors - merges,
        "merges": merges,
        "active_offsets": remaining_slots,
        "max_position_error_before_quantization": max_position_error,
        "mean_response_rms": float(np.mean(rms_response)) if merges else None,
        "mean_estimated_saving_bytes": float(np.mean(savings)) if merges else None,
        "expected_saving_min_bytes": min_saving_bytes,
        "candidate_neighbors": candidate_neighbors,
        "accepted_below_one_estimated_byte": int(np.count_nonzero(np.asarray(savings) < 1)),
        "accepted_nonpositive_estimated_saving": int(np.count_nonzero(np.asarray(savings) <= 0)),
        "response_rms_max": response_rms_limit,
        "rounds": history,
    }
    return arrays, rows, np.flatnonzero(live & merged), report


def select_camera_centers(train_cameras):
    """Use up to eight evenly spaced training cameras."""
    indices = np.linspace(0, len(train_cameras) - 1, min(8, len(train_cameras)))
    cameras = [train_cameras[i] for i in indices.round().astype(int)]
    centers = np.stack([camera.c2w[:3, 3] for camera in cameras]).astype(np.float32)
    return centers, [camera.image_name for camera in cameras]


@torch.no_grad()
def merge_training_anchors(model, train_cameras, result_dir, *,
                           response_rms_limit=RESPONSE_RMS_LIMIT,
                           min_saving_bytes=EXPECTED_SAVING_BYTES, candidate_neighbors=None):
    """Measure current coding cost and apply one merge event to the live optimizer."""
    core = model.core
    cameras, names = select_camera_centers(train_cameras)
    state = {
        key: value.detach().cpu()
        for key, value in core.state_dict().items()
        if key in ANCHOR_FIELDS or key.startswith(("mlp_color.", "mlp_opacity."))
    }
    coded = int(core.get_mask_anchor.detach().sum().item())
    if coded < 2:
        metadata = {"num_anchors": coded, "bit_offsets": 0}
    else:
        with TemporaryDirectory(prefix="anchor_merge_", dir=result_dir) as directory:
            metadata = model.encode_attributes(Path(directory))
    arrays, keep, merged, report = plan_anchor_merge(
        state, model.voxel_size, metadata, cameras,
        response_rms_limit=response_rms_limit,
        min_saving_bytes=min_saving_bytes,
        candidate_neighbors=candidate_neighbors,
    )
    report["selection_cameras"] = names
    if not len(merged):
        return report

    updated_rows = torch.as_tensor(merged, device=model.device)
    keep_mask = torch.zeros(model.num_anchors, dtype=torch.bool, device=model.device)
    keep_mask[torch.as_tensor(keep, device=model.device)] = True
    for name, array in arrays.items():
        parameter = getattr(core, name)
        parameter.index_copy_(
            0, updated_rows, torch.as_tensor(array[merged], device=model.device)
        )
        for value in model.optimizer.state.get(parameter, {}).values():
            if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                value.index_fill_(0, updated_rows, 0)

    # Reset histories tied to the old features or the old position.
    for name in (
        "sensitivity_feat", "sensitivity_scaling", "sensitivity_offsets",
        "sensitivity_sq_feat", "sensitivity_sq_scaling", "sensitivity_sq_offsets",
        "bits_ema", "coverage_ema", "mini_splat_importance", "semantic_target",
        "semantic_cov", "spa_z", "spa_u", "opacity_accum", "anchor_demon",
        "max_radii2D",
    ):
        buffer = getattr(core, name)
        if buffer.ndim and buffer.shape[0] == len(keep_mask):
            buffer.index_fill_(0, updated_rows, 0)
    for name in ("offset_gradient_accum", "offset_denom"):
        buffer = getattr(core, name)
        if buffer.shape[0] == len(keep_mask) * core.n_offsets:
            buffer.view(len(keep_mask), core.n_offsets, -1).index_fill_(0, updated_rows, 0)
    if core.max_radii2D.shape[0] == len(keep_mask):
        core.max_radii2D = core.max_radii2D[keep_mask]
    core.prune_anchor(~keep_mask, preserve_parameters=True)
    return report
