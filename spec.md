## Environment

A genetic algorithm where fitness is determined by a game between randomly paired organisms from the population.

## World

- 2D grid, `W = 2*org_cols + field_width` wide and `grid_height` tall, black/white cells
- Default: 46 wide (3 + 40 + 3) and 16 tall
- Initially all white except the two organisms and the red block

## Organisms

- Each organism is a `org_cols × genome_rows` pattern placed at the left or right edge, vertically from row 0
- Left occupies columns 0–(org_cols-1); right occupies the symmetric columns on the right edge (horizontally mirrored)
- Genome: two integers per organism
  1. **Pattern genome** (`genome_rows * org_cols` bits): encodes the initial cell pattern
  2. **Rule genome** (18 bits): encodes a per-organism "near rule" (see below)

## CA Update Rule (far rule)

Game of Life-style 2D rule applied globally each step:

- Count live Moore neighbors for each cell (8-connected, excluding self)
- A dead cell with exactly `birth` neighbors becomes alive; a live cell with `survival` neighbors survives; otherwise it dies
- Default: birth=[3], survival=[2,3] (standard GoL)

## Red Block

- Starts at CENTER_COL = `org_cols + field_width//2`, CENTER_ROW = `grid_height//2`
- Rendered as always-live in the CA grid (so patterns can interact with it)
- Preserved each step regardless of CA outcome

Movement: after each step, if the CA rule would have killed the block cell, forces from neighboring live cells push it:
- For each of the 4 cardinal neighbors: if that neighbor was alive before the step, a force pushes the block away from it
- Forces sum; the block moves at most 1 cell per step (clamped to grid bounds)

## Near Rule

When the block contacts a pattern, a different evolvable rule activates for the 8 cells in the Moore neighborhood of the block:

- The 18-bit rule genome encodes birth (bits 0–8) and survival (bits 9–17) for neighbor counts 0–8
- Cells left of CENTER_COL use the **left organism's** near rule; cells right of CENTER_COL use the **right organism's** near rule
- This lets organisms evolve specialized behavior when they reach the block

## Scoring

- Left organism's score  = final_col − CENTER_COL  (positive = block moved right, away from left)
- Right organism's score = CENTER_COL − final_col  (positive = block moved left, away from right)

A game ends after `steps_per_game` steps, or early if the block reaches either organism's starting columns (column ≤ org_cols or column ≥ RIGHT_START_COL).

## Genetic Algorithm

- Population: N organisms, each with a pattern genome and a rule genome
- Each round: each organism plays G games against randomly selected opponents (random side each game)
- Score = mean score across all games
- Selection: linear rank-based sampling (CPU, to avoid MPS multinomial bugs)
- Reproduction: same parent selection applied to both genomes; vectorized bit-flip mutation at `mutation_rate`
- Elitism: top organism carried over unchanged each round
- Reference evaluation each round: all organisms vs. round-0 population and vs. 10-rounds-ago population

## Config (Hydra)

- `field_width`: int (default: 40)
- `grid_height`: int (default: 16)
- `org_cols`: int (default: 3)
- `genome_rows`: int (default: 16)
- `birth`: list[int] (default: [3])
- `survival`: list[int] (default: [2, 3])
- `mutation_rate`: float (default: 0.005)
- `crossover`: bool (default: false)
- `population_size`: int (default: 100)
- `games_per_round`: int (default: 20)
- `n_rounds`: int (default: 25)
- `steps_per_game`: int (default: 300)

## Output

- `rounds/{n}/` subfolder each round:
  - GIF of best, median, worst organism's best/worst/random game (score in filename)
  - GIF of median organism vs. round-0 and round-10-ago opponent
- `rounds/win_rates.png`: 3-panel score distribution plot (vs current, vs round 0, vs round -10), updated each round
