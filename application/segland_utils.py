import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def load_checkpoint_for_submodule(model, ckpt_path, submodule_name, map_location="cpu", strict=True):
    """Load checkpoint `ckpt_path` and apply weights that belong to `submodule_name`.

    This helper filters keys in the checkpoint `state_dict` to those that refer to
    the submodule (e.g. starting with "segmentor." or "tooth_model.") and strips
    that prefix before loading into `model.<submodule_name>`.

    Args:
        model: full model instance containing the submodule attribute.
        ckpt_path: path to checkpoint file (torch.save format).
        submodule_name: attribute name of the submodule on `model` (str).
        map_location: device mapping for torch.load.
        strict: passed to `load_state_dict`.
    Returns:
        dict: loaded state_dict keys that were applied (for logging)
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt.get("state_dict", ckpt)

    # Normalize keys: remove leading 'module.' if present
    normalized = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        normalized[nk] = v

    # Determine accepted prefixes for this submodule
    prefixes = [f"{submodule_name}.", submodule_name + "."]

    filtered = {}
    for k, v in normalized.items():
        for p in prefixes:
            if k.startswith(p):
                newk = k[len(p) :]
                filtered[newk] = v
                break

    submodule = getattr(model, submodule_name, None)
    if submodule is None:
        raise AttributeError(f"Model has no submodule named '{submodule_name}'")

    if len(filtered) == 0:
        # Try to match keys without prefix (checkpoint saved from submodule-only training)
        candidate = {}
        for k, v in normalized.items():
            candidate[k] = v
        try:
            submodule.load_state_dict(candidate, strict=strict)
            return list(candidate.keys())
        except Exception:
            # nothing matched
            return []

    # Load filtered into submodule
    submodule.load_state_dict(filtered, strict=strict)
    return list(filtered.keys())


# --- Heatmap -> landmark extraction helpers (minimal, adapted from HeatmapTesterV2) ---
def extract_single_point_proposal(verts, channel_pred, percentile=95):
    thr = np.percentile(channel_pred, percentile)
    mask = channel_pred >= thr
    if mask.sum() == 0:
        # fallback: take global max
        idx = int(np.argmax(channel_pred))
        return [tuple(verts[idx])]
    pts = verts[mask]
    centroid = pts.mean(axis=0)
    # return nearest vertex to centroid
    dists = np.linalg.norm(verts - centroid[None, :], axis=1)
    idx = int(np.argmin(dists))
    return [tuple(verts[idx])]


def extract_multi_point_proposal(verts, channel_pred, k=4, percentile=95):
    thr = np.percentile(channel_pred, percentile)
    mask = channel_pred >= thr
    if mask.sum() == 0:
        # fallback: top-k peaks
        idxs = np.argsort(channel_pred)[-k:][::-1]
        return [tuple(verts[i]) for i in idxs]
    pts = verts[mask]
    weights = channel_pred[mask]
    if pts.shape[0] <= k:
        # fewer points than clusters: return all
        return [tuple(p) for p in pts]
    kmeans = KMeans(n_clusters=k, random_state=0).fit(pts, sample_weight=weights)
    centers = kmeans.cluster_centers_
    # map centers to nearest vertices
    proposals = []
    for c in centers:
        dists = np.linalg.norm(verts - c[None, :], axis=1)
        idx = int(np.argmin(dists))
        proposals.append(tuple(verts[idx]))
    return proposals


def _get_num_clusters_from_components(mesh_edges, inds, min_k=2, max_k=6):
    # mesh_edges: array (E,2) of vertex indices
    # inds: indices of vertices belonging to connected component
    if len(inds) == 0:
        return 0
    # build adjacency for subgraph
    mask = np.isin(mesh_edges, inds)
    # select edges where both ends in inds
    both = np.logical_and(mask[:, 0], mask[:, 1]) if mask.ndim > 1 else np.array([False])
    if isinstance(both, np.ndarray) and both.sum() == 0:
        return min_k
    edges_sub = mesh_edges[both]
    # remap vertex ids to small indices
    mapping = {v: i for i, v in enumerate(inds)}
    rows = [mapping[e[0]] for e in edges_sub]
    cols = [mapping[e[1]] for e in edges_sub]
    data = np.ones(len(rows), dtype=np.int8)
    n = len(inds)
    if n == 0:
        return min_k
    csr = csr_matrix((data, (rows, cols)), shape=(n, n))
    n_comp, labels = connected_components(csgraph=csr, directed=False, connection='weak')
    k = int(np.clip(n_comp, min_k, max_k))
    return k


def extract_variable_point_proposal(mesh_edges, verts, channel_pred, min_k=2, max_k=6, percentile=95):
    thr = np.percentile(channel_pred, percentile)
    mask = channel_pred >= thr
    if mask.sum() == 0:
        return []
    inds = np.where(mask)[0]
    k = _get_num_clusters_from_components(mesh_edges, inds, min_k, max_k)
    if k <= 0:
        return []
    return extract_multi_point_proposal(verts, channel_pred, k=k, percentile=percentile)


def mesial_distal_correction(channels_proposals: dict):
    # placeholder: SDK correction implemented in HeatmapTesterV2; here we keep identity
    return channels_proposals


def extract_landmarks_from_heatmaps(heatmaps, tooth_meshes, full_path=""):
    """heatmaps: list of arrays (num_channels, N_i) per tooth
       tooth_meshes: list of dicts with 'coord' arrays
       Returns dict mapping tooth_idx -> channel -> list of proposals
    """
    all_proposals = {}
    for ti, (hm, tooth) in enumerate(zip(heatmaps, tooth_meshes)):
        verts = np.asarray(tooth['coord'])
        if verts.ndim != 2:
            continue
        proposals = {}
        # detect heatmap shape: (channels, N) or (N, channels)
        hm_arr = np.asarray(hm)
        if hm_arr.ndim == 1:
            # single-channel vector per-vertex -> treat as (N, 1)
            hm_arr = hm_arr[:, None]

        if hm_arr.shape[0] == verts.shape[0]:
            # shape is (N, C)
            num_channels = hm_arr.shape[1]
            for ch in range(num_channels):
                channel_pred = hm_arr[:, ch]
                # choose method depending on channel: here we approximate
                if ch in (0, 1, 2, 4, 5, 7, 8, 9):
                    proposals[ch] = extract_single_point_proposal(verts, channel_pred)
                elif ch == 3:
                    proposals[ch] = extract_multi_point_proposal(verts, channel_pred, k=4)
                elif ch == 6:
                    mesh_edges = np.empty((0, 2), dtype=int)
                    proposals[ch] = extract_variable_point_proposal(mesh_edges, verts, channel_pred)
                else:
                    proposals[ch] = []
        else:
            # assume shape is (C, N)
            for ch in range(hm_arr.shape[0]):
                channel_pred = hm_arr[ch]
            # choose method depending on channel: here we approximate
                if ch in (0, 1, 2, 4, 5, 7, 8, 9):
                    proposals[ch] = extract_single_point_proposal(verts, channel_pred)
                elif ch == 3:
                    proposals[ch] = extract_multi_point_proposal(verts, channel_pred, k=4)
                elif ch == 6:
                    # variable cusps
                    mesh_edges = np.empty((0, 2), dtype=int)
                    proposals[ch] = extract_variable_point_proposal(mesh_edges, verts, channel_pred)
                else:
                    proposals[ch] = []
        proposals = mesial_distal_correction(proposals)
        all_proposals[ti] = proposals
    return all_proposals
