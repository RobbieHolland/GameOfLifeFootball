import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn.functional as F

ORG_COLS = 3
FIELD_WIDTH = 40
W = 2 * ORG_COLS + FIELD_WIDTH
H = 16
CENTER_COL = ORG_COLS + FIELD_WIDTH // 2
CENTER_ROW = H // 2
RIGHT_START_COL = ORG_COLS + FIELD_WIDTH

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def configure(field_width, h, org_cols=3):
    global W, H, ORG_COLS, FIELD_WIDTH, CENTER_COL, CENTER_ROW, RIGHT_START_COL, _NEIGHBOR_KERNEL
    ORG_COLS = org_cols
    FIELD_WIDTH = field_width
    W = 2 * org_cols + field_width
    H = h
    CENTER_COL = org_cols + field_width // 2
    CENTER_ROW = H // 2
    RIGHT_START_COL = org_cols + field_width
    _NEIGHBOR_KERNEL = None


_NEIGHBOR_KERNEL = None


def _get_kernel():
    global _NEIGHBOR_KERNEL
    if _NEIGHBOR_KERNEL is None:
        k = torch.ones(1, 1, 3, 3, device=DEVICE)
        k[0, 0, 1, 1] = 0
        _NEIGHBOR_KERNEL = k
    return _NEIGHBOR_KERNEL


def build_2d_rule(birth, survival):
    birth_mask = torch.zeros(9, dtype=torch.bool, device=DEVICE)
    survival_mask = torch.zeros(9, dtype=torch.bool, device=DEVICE)
    for b in birth:
        birth_mask[b] = True
    for s in survival:
        survival_mask[s] = True
    return birth_mask, survival_mask


def step_games(grids, red_cols, red_rows, birth_mask, survival_mask,
               near_birth_mask, near_surv_mask):
    B = grids.shape[0]
    kernel = _get_kernel()
    batch_idx = torch.arange(B, device=DEVICE)
    rc, rr = red_cols, red_rows

    grids = grids.clone()
    grids[batch_idx, rr, rc] = 1

    nc = F.conv2d(grids.float().unsqueeze(1), kernel, padding=1).squeeze(1).long()

    # Far rule (global)
    far_born = (grids == 0) & birth_mask[nc]
    far_survive = (grids == 1) & survival_mask[nc]

    # Adjacency mask: Moore neighborhood of red block
    rows = torch.arange(H, device=DEVICE)[None, :, None]
    cols = torch.arange(W, device=DEVICE)[None, None, :]
    rr_e = rr[:, None, None]
    rc_e = rc[:, None, None]
    adj_mask = ((rows - rr_e).abs() <= 1) & ((cols - rc_e).abs() <= 1)
    adj_mask &= ~((rows == rr_e) & (cols == rc_e))  # exclude block cell

    near_born    = near_birth_mask[nc]   # (B, H, W)
    near_survive = near_surv_mask[nc]

    near_result = torch.where(grids == 0, near_born, near_survive).to(torch.uint8)
    far_result  = (far_born | far_survive).to(torch.uint8)
    new_grids = torch.where(adj_mask, near_result, far_result)

    # Movement
    ca_kills_block = new_grids[batch_idx, rr, rc] == 0
    new_grids[batch_idx, rr, rc] = 1

    delta_col = torch.zeros(B, dtype=torch.int32, device=DEVICE)
    delta_row = torch.zeros(B, dtype=torch.int32, device=DEVICE)

    for adj_dr, adj_dc, fdc, fdr in [(0, 1, -1, 0), (0, -1, 1, 0), (1, 0, 0, -1), (-1, 0, 0, 1)]:
        adj_r = rr + adj_dr
        adj_c = rc + adj_dc
        valid = (adj_r >= 0) & (adj_r < H) & (adj_c >= 0) & (adj_c < W)
        cond1 = grids[batch_idx, adj_r.clamp(0, H-1), adj_c.clamp(0, W-1)] == 1
        force = valid & cond1 & ca_kills_block
        delta_col += force.int() * fdc
        delta_row += force.int() * fdr

    new_red_cols = (rc + delta_col).clamp(0, W - 1)
    new_red_rows = (rr + delta_row).clamp(0, H - 1)

    return new_grids, new_red_cols, new_red_rows


def init_grids(left_genomes, right_genomes, genome_rows):
    B = left_genomes.shape[0]
    grids = torch.zeros(B, H, W, dtype=torch.uint8, device=DEVICE)
    n_bits = genome_rows * ORG_COLS

    for b in range(B):
        lg = left_genomes[b].item()
        rg = right_genomes[b].item()
        for bit in range(n_bits):
            r = bit // ORG_COLS
            c = bit % ORG_COLS
            grids[b, r, c] = (lg >> bit) & 1
        for bit in range(n_bits):
            r = bit // ORG_COLS
            mirrored_col = (ORG_COLS - 1) - (bit % ORG_COLS)
            grids[b, r, RIGHT_START_COL + mirrored_col] = (rg >> bit) & 1

    return grids


def run_games(left_patterns, right_patterns, birth_mask, survival_mask,
              near_birth_mask, near_surv_mask, steps, genome_rows, record=False):
    B = left_patterns.shape[0]
    grids = init_grids(left_patterns, right_patterns, genome_rows)
    red_cols = torch.full((B,), CENTER_COL, dtype=torch.int32, device=DEVICE)
    red_rows = torch.full((B,), CENTER_ROW, dtype=torch.int32, device=DEVICE)

    active = torch.ones(B, dtype=torch.bool, device=DEVICE)
    final_cols = red_cols.clone()

    frames = [grids.cpu().clone()] if record else None
    red_positions = [(red_cols.cpu().clone(), red_rows.cpu().clone())] if record else None

    for _ in range(steps):
        if not active.any():
            break
        new_grids, new_red_cols, new_red_rows = step_games(
            grids, red_cols, red_rows, birth_mask, survival_mask,
            near_birth_mask, near_surv_mask)

        ended = active & ((new_red_cols <= ORG_COLS) | (new_red_cols >= RIGHT_START_COL))
        final_cols = torch.where(ended, new_red_cols, final_cols)
        active = active & ~ended

        grids = new_grids
        red_cols = new_red_cols
        red_rows = new_red_rows

        if record:
            frames.append(grids.cpu().clone())
            red_positions.append((red_cols.cpu().clone(), red_rows.cpu().clone()))

    final_cols = torch.where(active, red_cols, final_cols)

    left_scores  = (final_cols - CENTER_COL).float()
    right_scores = (CENTER_COL - final_cols).float()

    if record:
        return left_scores, right_scores, frames, red_positions
    return left_scores, right_scores
