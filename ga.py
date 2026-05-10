import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sim import build_2d_rule, run_games, configure, DEVICE


def evaluate_population(genomes: torch.Tensor, birth_mask, survival_mask,
                        near_birth_mask, near_surv_mask, cfg):
    N = genomes.shape[0]
    G = cfg.games_per_round
    scores = torch.zeros(N, device=DEVICE)

    org_indices      = torch.randint(0, N, (N * G,), device=DEVICE)
    opponent_indices = torch.randint(0, N, (N * G,), device=DEVICE)
    sides            = torch.randint(0, 2, (N * G,), device=DEVICE)

    left_genomes  = torch.where(sides == 0, genomes[org_indices], genomes[opponent_indices])
    right_genomes = torch.where(sides == 1, genomes[org_indices], genomes[opponent_indices])

    left_s, right_s = run_games(
        left_genomes, right_genomes,
        birth_mask, survival_mask, near_birth_mask, near_surv_mask,
        cfg.steps_per_game, cfg.genome_rows)

    org_scores = torch.where(sides == 0, left_s, right_s)
    scores.scatter_add_(0, org_indices, org_scores)
    counts = torch.zeros(N, device=DEVICE)
    counts.scatter_add_(0, org_indices, torch.ones(N * G, device=DEVICE))
    mean_scores = scores / counts.clamp(min=1)

    org_indices_cpu   = org_indices.cpu()
    org_scores_cpu    = org_scores.cpu()
    left_genomes_cpu  = left_genomes.cpu()
    right_genomes_cpu = right_genomes.cpu()
    sides_cpu         = sides.cpu()

    best_left   = torch.zeros(N, dtype=torch.long)
    best_right  = torch.zeros(N, dtype=torch.long)
    best_side   = torch.zeros(N, dtype=torch.long)
    worst_left  = torch.zeros(N, dtype=torch.long)
    worst_right = torch.zeros(N, dtype=torch.long)
    worst_side  = torch.zeros(N, dtype=torch.long)
    best_score  = torch.full((N,), float('-inf'))
    worst_score = torch.full((N,), float('inf'))

    for i in range(N * G):
        org = org_indices_cpu[i].item()
        s   = org_scores_cpu[i].item()
        if s > best_score[org]:
            best_score[org]  = s
            best_left[org]   = left_genomes_cpu[i]
            best_right[org]  = right_genomes_cpu[i]
            best_side[org]   = sides_cpu[i]
        if s < worst_score[org]:
            worst_score[org]  = s
            worst_left[org]   = left_genomes_cpu[i]
            worst_right[org]  = right_genomes_cpu[i]
            worst_side[org]   = sides_cpu[i]

    return mean_scores, best_left, best_right, best_side, worst_left, worst_right, worst_side


def select_and_reproduce(genomes: torch.Tensor, scores: torch.Tensor, cfg) -> torch.Tensor:
    N = genomes.shape[0]

    ranks = scores.cpu().argsort().argsort().float()
    probs = (ranks + 1) / ((N * (N + 1)) / 2)
    probs = probs / probs.sum()

    parents_idx    = torch.multinomial(probs, N, replacement=True)
    selected_ranks = ranks[parents_idx].long().tolist()
    rank_counts    = torch.zeros(N, dtype=torch.long)
    for r in selected_ranks:
        rank_counts[r] += 1
    top10 = rank_counts[-10:].flip(0).tolist()
    print(f"  selected rank counts (top10→worst): {top10}")
    shuffle      = torch.randperm(N)
    parents_idx  = parents_idx[shuffle]
    parents      = genomes.cpu()[parents_idx].to(DEVICE)

    n_bits      = cfg.genome_rows * cfg.org_cols
    genome_mask = (1 << n_bits) - 1

    if cfg.crossover:
        half = N // 2
        p1, p2      = parents[:half], parents[half:half * 2]
        cross_point = torch.randint(1, n_bits, (half,), device=DEVICE)
        offspring   = torch.zeros(N, dtype=torch.long, device=DEVICE)
        for i in range(half):
            cp   = cross_point[i].item()
            mask = (1 << cp) - 1
            offspring[i * 2]     = (p1[i].item() & mask) | (p2[i].item() & ~mask & genome_mask)
            offspring[i * 2 + 1] = (p2[i].item() & mask) | (p1[i].item() & ~mask & genome_mask)
    else:
        offspring = parents.clone()

    bit_values = torch.tensor([1 << b for b in range(n_bits)], dtype=torch.long)
    flip       = torch.rand(N, n_bits) < cfg.mutation_rate
    flip_masks = (flip.long() * bit_values).sum(dim=1).to(DEVICE)
    offspring  = (offspring ^ flip_masks) & genome_mask

    return offspring


def evaluate_vs_population(genomes: torch.Tensor, ref_genomes: torch.Tensor,
                            birth_mask, survival_mask, near_birth_mask, near_surv_mask,
                            cfg) -> np.ndarray:
    N         = genomes.shape[0]
    ref_idx   = torch.randint(0, len(ref_genomes), (N,))
    opponents = ref_genomes[ref_idx].to(DEVICE)
    sides     = torch.randint(0, 2, (N,), device=DEVICE)
    left_g    = torch.where(sides == 0, genomes,   opponents)
    right_g   = torch.where(sides == 1, genomes,   opponents)
    left_s, right_s = run_games(
        left_g, right_g,
        birth_mask, survival_mask, near_birth_mask, near_surv_mask,
        cfg.steps_per_game, cfg.genome_rows)
    scores = torch.where(sides == 0, left_s, right_s)
    return scores.cpu().numpy()


def record_game(genome_left: int, genome_right: int,
                birth_mask, survival_mask, near_birth_mask, near_surv_mask, cfg):
    lg = torch.tensor([genome_left],  device=DEVICE)
    rg = torch.tensor([genome_right], device=DEVICE)
    ls, rs, frames, red_pos = run_games(
        lg, rg, birth_mask, survival_mask, near_birth_mask, near_surv_mask,
        cfg.steps_per_game, cfg.genome_rows, record=True)
    return frames, red_pos, ls[0].item(), rs[0].item()


def save_gif(frames, red_positions, path: Path, genome_label: str):
    import imageio
    images = []
    scale  = 10
    for grid, (rcols, rrows) in zip(frames, red_positions):
        g = grid[0].numpy()
        gh, gw = g.shape
        img = np.ones((gh * scale, gw * scale, 3), dtype=np.uint8) * 255
        ys, xs = np.where(g == 1)
        for y, x in zip(ys, xs):
            img[y*scale:(y+1)*scale, x*scale:(x+1)*scale] = 0
        rc = rcols[0].item()
        rr = rrows[0].item()
        img[rr*scale:(rr+1)*scale, rc*scale:(rc+1)*scale] = [255, 0, 0]
        images.append(img)
    imageio.mimsave(str(path), images, fps=20)


def _plot_deciles(ax, all_scores, title):
    rounds  = list(range(len(all_scores)))
    deciles = np.array([[np.percentile(s, q) for q in range(10, 100, 10)] for s in all_scores])
    for i, q in enumerate(range(10, 100, 10)):
        lw  = 2 if q == 50 else 1
        lbl = f"p{q}" if q in (10, 50, 90) else None
        ax.plot(rounds, deciles[:, i], linewidth=lw, label=lbl)
    ax.set_xlabel("Round")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()


def save_score_plot(all_scores, all_ref0_scores, all_ref10_scores, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    _plot_deciles(axes[0], all_scores, "vs current population")
    if all_ref0_scores:
        _plot_deciles(axes[1], all_ref0_scores, "vs round 0 population")
    if all_ref10_scores:
        _plot_deciles(axes[2], all_ref10_scores, "vs population 10 rounds ago")

    all_combined = all_scores + all_ref0_scores + all_ref10_scores
    ymin = min(np.min(s) for s in all_combined)
    ymax = max(np.max(s) for s in all_combined)
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    fig.savefig(str(out_dir / "win_rates.png"), dpi=100)
    plt.close(fig)


def run_ga(cfg):
    configure(cfg.field_width, cfg.grid_height, cfg.org_cols)
    N = cfg.population_size
    birth_mask, survival_mask         = build_2d_rule(list(cfg.birth),       list(cfg.survival))
    near_birth_mask, near_surv_mask   = build_2d_rule(list(cfg.near_birth),  list(cfg.near_survival))

    n_bits     = cfg.genome_rows * cfg.org_cols
    genome_max = 1 << n_bits
    genomes    = torch.randint(0, genome_max, (N,), dtype=torch.long, device=DEVICE)

    all_scores       = []
    all_ref0_scores  = []
    all_ref10_scores = []
    population_history = []

    for round_num in range(cfg.n_rounds):
        print(f"Round {round_num + 1}/{cfg.n_rounds}")
        (scores,
         best_left, best_right, best_side,
         worst_left, worst_right, worst_side
        ) = evaluate_population(genomes, birth_mask, survival_mask,
                                near_birth_mask, near_surv_mask, cfg)
        all_scores.append(scores.cpu().numpy())

        sorted_idx = scores.argsort()
        n          = genomes.shape[0]
        best_idx   = sorted_idx[-1].item()
        median_idx = sorted_idx[n // 2].item()
        worst_idx  = sorted_idx[0].item()

        sample_idx = torch.randperm(n)[:100]
        population_history.append(genomes[sample_idx].cpu().clone())

        ref0_g = population_history[0].to(DEVICE)
        ref0   = evaluate_vs_population(genomes, ref0_g, birth_mask, survival_mask,
                                        near_birth_mask, near_surv_mask, cfg)
        all_ref0_scores.append(ref0)

        ref10_round  = max(0, round_num - 10)
        ref10_g      = population_history[ref10_round].to(DEVICE)
        ref10        = evaluate_vs_population(genomes, ref10_g, birth_mask, survival_mask,
                                              near_birth_mask, near_surv_mask, cfg)
        all_ref10_scores.append(ref10)

        print(f"  best={scores[best_idx]:.2f} median={scores[median_idx]:.2f} worst={scores[worst_idx]:.2f} | vs_r0 med={np.median(ref0):.2f} vs_r10 med={np.median(ref10):.2f}")

        round_dir = Path(f"rounds/{round_num}")
        round_dir.mkdir(parents=True, exist_ok=True)

        if round_num % cfg.save_gifs_interval == 0:
            for label, idx, game_src in [("best", best_idx, "best"), ("median", median_idx, "random"), ("worst", worst_idx, "worst")]:
                if game_src == "best":
                    lg, rg = best_left[idx].item(), best_right[idx].item()
                    side   = "left" if best_side[idx].item() == 0 else "right"
                elif game_src == "worst":
                    lg, rg = worst_left[idx].item(), worst_right[idx].item()
                    side   = "left" if worst_side[idx].item() == 0 else "right"
                else:
                    g       = genomes[idx].item()
                    opp_idx = sorted_idx[torch.randint(n // 2, n, (1,)).item()].item()
                    opp     = genomes[opp_idx].item()
                    side    = "left" if torch.rand(1).item() < 0.5 else "right"
                    lg, rg  = (g, opp) if side == "left" else (opp, g)
                frames, red_pos, ls, rs = record_game(lg, rg, birth_mask, survival_mask,
                                                      near_birth_mask, near_surv_mask, cfg)
                score = ls if side == "left" else rs
                save_gif(frames, red_pos, round_dir / f"{label}_{side}_{score:.2g}.gif", label)

            med_genome = genomes[median_idx].item()
            for ref_label, ref_pop in [("vs_r0", population_history[0]),
                                       ("vs_r10", population_history[ref10_round])]:
                opp_i  = torch.randint(len(ref_pop), (1,)).item()
                opp    = ref_pop[opp_i].item()
                side   = "left" if torch.rand(1).item() < 0.5 else "right"
                lg, rg = (med_genome, opp) if side == "left" else (opp, med_genome)
                frames, red_pos, ls, rs = record_game(lg, rg, birth_mask, survival_mask,
                                                      near_birth_mask, near_surv_mask, cfg)
                score = ls if side == "left" else rs
                save_gif(frames, red_pos, round_dir / f"median_{ref_label}_{side}_{score:.2g}.gif", ref_label)

        save_score_plot(all_scores, all_ref0_scores, all_ref10_scores, Path("rounds"))

        genomes = select_and_reproduce(genomes, scores, cfg)

    print("Done.")
