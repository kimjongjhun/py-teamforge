# balancer/team_balancer.py
import math
import time
import os
import heapq
import logging
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from ortools.sat.python import cp_model
from models.player import Player

# ---------------------------
# Logging configuration (console + file)
# ---------------------------
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "teamforge.log")
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("team_balancer")
logger.setLevel(logging.INFO)

# Console handler (info-level by default, debug will also go through depending on formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch_formatter = logging.Formatter("[%(levelname)s] %(message)s")
ch.setFormatter(ch_formatter)

# File handler (debug-level)
fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setLevel(logging.INFO)
fh_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
fh.setFormatter(fh_formatter)

# Avoid duplicate handlers if reloaded
if not logger.handlers:
    logger.addHandler(ch)
    logger.addHandler(fh)
else:
    # Replace handlers (safe in interactive reload)
    logger.handlers = []
    logger.addHandler(ch)
    logger.addHandler(fh)


# ---------------------------
# Helpers: spread evaluation
# ---------------------------
def evaluate_spread(teams: List[List[Player]]) -> float:
    totals = [sum(p.average for p in t) for t in teams if len(t) > 0]
    if not totals:
        return math.inf
    return max(totals) - min(totals)


# ---------------------------
# Team-size distributions (4s, up to two 3s, or one 5)
# ---------------------------
def compute_team_sizes(n: int) -> List[List[int]]:
    layouts: List[List[int]] = []

    # all 4s
    if n % 4 == 0:
        layouts.append([4] * (n // 4))

    # one 5 + rest 4s
    if n >= 5 and (n - 5) % 4 == 0:
        layouts.append([5] + [4] * ((n - 5) // 4))

    # one 3 + rest 4s
    if n >= 3 and (n - 3) % 4 == 0:
        layouts.append([3] + [4] * ((n - 3) // 4))

    # two 3s + rest 4s
    if n >= 6 and (n - 6) % 4 == 0:
        layouts.append([3, 3] + [4] * ((n - 6) // 4))

    logger.debug("Computed layouts for %d players: %s", n, layouts)
    return layouts


# ---------------------------
# Build friend blocks; split only when block > max_capacity
# ---------------------------
def build_blocks(players: List[Player], max_capacity: int) -> List[List[Player]]:
    friend_map: Dict[str, List[Player]] = defaultdict(list)
    singles: List[Player] = []

    for p in players:
        key = p.friend_code.strip() if p.friend_code else None
        if key:
            friend_map[key].append(p)
        else:
            singles.append(p)

    blocks: List[List[Player]] = []

    for grp_key, grp in friend_map.items():
        sorted_grp = sorted(grp, key=lambda x: -x.average)
        if len(sorted_grp) <= max_capacity:
            blocks.append(sorted_grp)
            logger.debug("Friend block kept as-is (key=%s, size=%d)", grp_key, len(sorted_grp))
        else:
            # split into chunks sized max_capacity
            logger.debug(
                "Friend block (key=%s) larger than max_capacity (%d). Splitting into chunks.",
                grp_key,
                max_capacity,
            )
            while sorted_grp:
                chunk = sorted_grp[:max_capacity]
                blocks.append(chunk)
                sorted_grp = sorted_grp[max_capacity:]

    # add singles as single-element blocks
    for s in singles:
        blocks.append([s])

    # sort blocks: larger blocks first (reduces backtracking), tie-break by block avg desc
    blocks.sort(key=lambda b: (-len(b), -sum(p.average for p in b)))
    logger.debug("Built %d blocks (max_capacity=%d). Blocks sizes: %s", len(blocks), max_capacity, [len(b) for b in blocks])
    return blocks


# ---------------------------
# Constraint checks: family and group (group per-team limit is parameterized)
# ---------------------------
def violates_family(team: List[Player], block: List[Player]) -> bool:
    existing = {p.family_code for p in team if p.family_code}
    for p in block:
        if p.family_code and p.family_code in existing:
            return True
    return False


def violates_group_with_limit(team: List[Player], block: List[Player], max_per_group: int) -> bool:
    """
    Return True if adding 'block' to 'team' would exceed max_per_group for any group_code.
    max_per_group is >=1 (strict mode) or 2 (relaxed).
    """
    if max_per_group < 1:
        max_per_group = 1
    existing_counts: Dict[str, int] = {}
    for p in team:
        if p.group_code:
            existing_counts[p.group_code] = existing_counts.get(p.group_code, 0) + 1
    for p in block:
        if not p.group_code:
            continue
        cnt = existing_counts.get(p.group_code, 0) + 1
        if cnt > max_per_group:
            return True
        existing_counts[p.group_code] = cnt
    return False


# ---------------------------
# Optimistic spread estimator used for pruning during backtracking.
# Greedily assigns remaining players (flattened) to lowest-total eligible teams.
# ---------------------------
def optimistic_spread_if_fill(teams: List[List[Player]], remaining_blocks: List[List[Player]], sizes: List[int]) -> float:
    totals = [sum(p.average for p in t) for t in teams]
    caps = [sizes[i] - len(teams[i]) for i in range(len(sizes))]

    remaining_avgs: List[float] = []
    for blk in remaining_blocks:
        for p in blk:
            remaining_avgs.append(p.average)
    remaining_avgs.sort(reverse=True)

    # min-heap of (current_total, team_index) only for teams with capacity >0
    heap: List[tuple] = []
    for i, cap in enumerate(caps):
        if cap > 0:
            heap.append((totals[i], i))
    heapq.heapify(heap)

    caps_copy = caps[:]
    totals_copy = totals[:]
    for avg in remaining_avgs:
        if not heap:
            break
        total, ti = heapq.heappop(heap)
        totals_copy[ti] += avg
        caps_copy[ti] -= 1
        if caps_copy[ti] > 0:
            heapq.heappush(heap, (totals_copy[ti], ti))

    if not totals_copy:
        return math.inf
    return max(totals_copy) - min(totals_copy)


# ---------------------------
# Backtracking assignment with optimistic pruning and timeout
# ---------------------------
def try_assign_with_timeout(
    players: List[Player],
    sizes: List[int],
    max_group_per_team: int = 1,
    timeout_seconds: float = 10.0,
) -> List[List[Player]]:
    start_time = time.time()
    max_capacity = max(sizes)
    blocks = build_blocks(players, max_capacity)

    # Quick infeasibility check: any block larger than largest size -> impossible
    if any(len(b) > max_capacity for b in blocks):
        logger.debug("Layout impossible: a block is larger than max_capacity=%d", max_capacity)
        return []

    teams: List[List[Player]] = [[] for _ in sizes]
    n_blocks = len(blocks)

    # precompute remaining players from idx for capacity pruning
    remaining_players_from = [0] * (n_blocks + 1)
    for i in range(n_blocks - 1, -1, -1):
        remaining_players_from[i] = remaining_players_from[i + 1] + len(blocks[i])

    def backtrack(idx: int) -> bool:
        # timeout check
        if time.time() - start_time > timeout_seconds:
            logger.debug("try_assign timed out after %.2fs (idx=%d)", time.time() - start_time, idx)
            return False

        # capacity feasibility prune
        remaining = remaining_players_from[idx]
        free_slots = sum(sizes[i] - len(teams[i]) for i in range(len(sizes)))
        if remaining > free_slots:
            logger.debug("Pruned by capacity: remaining players=%d > free_slots=%d at idx=%d", remaining, free_slots, idx)
            return False

        if idx == n_blocks:
            # all placed
            logger.debug("All blocks placed successfully (time=%.2fs)", time.time() - start_time)
            return True

        block = blocks[idx]
        logger.debug("Placing block %d/%d (size=%d, avg=%.2f)", idx + 1, n_blocks, len(block), sum(p.average for p in block) / len(block))

        # heuristic: try teams in order of increasing total (attempt to keep balanced)
        order = sorted(range(len(teams)), key=lambda i: sum(p.average for p in teams[i]))

        for ti in order:
            if len(teams[ti]) + len(block) > sizes[ti]:
                logger.debug("  team %d rejected: capacity (%d/%d) would overflow", ti, len(teams[ti]) + len(block), sizes[ti])
                continue
            if violates_family(teams[ti], block):
                logger.debug("  team %d rejected: family conflict", ti)
                continue
            if violates_group_with_limit(teams[ti], block, max_group_per_team):
                logger.debug("  team %d rejected: group conflict against max_per_group=%d", ti, max_group_per_team)
                continue

            # tentatively place
            teams[ti].extend(block)
            logger.debug("  placed on team %d (now size %d).", ti, len(teams[ti]))

            # optimistic pruning: compute minimal achievable spread if we greedily fill remaining blocks
            optimistic = optimistic_spread_if_fill(teams, blocks[idx + 1:], sizes)
            logger.debug("    optimistic spread if filled = %.2f", optimistic)

            # don't prune aggressively using spread here; main loop handles spread threshold.
            if backtrack(idx + 1):
                return True

            # undo
            for _ in block:
                teams[ti].pop()
            logger.debug("  backtracked from team %d", ti)

        return False

    ok = backtrack(0)
    return teams if ok else []


# ---------------------------
# Optimization pass: time-limited 1-for-1 singleton swaps.
# Does not break friend groups (only swaps players without friend_code).
# ---------------------------
def optimize_balance_with_timeout(teams: List[List[Player]], timeout_seconds: float = 10.0) -> List[List[Player]]:
    start = time.time()

    def can_swap(p: Player, q: Player, t1: List[Player], t2: List[Player]) -> bool:
        # do not break friend groups: only allow swap if both players have no friend_code
        if p.friend_code or q.friend_code:
            return False

        # after swap, families and group counts must hold
        fam1 = {m.family_code for m in t1 if m is not p and m.family_code}
        grp1 = {m.group_code for m in t1 if m is not p and m.group_code}
        if q.family_code and q.family_code in fam1:
            return False
        if q.group_code and q.group_code in grp1:
            return False

        fam2 = {m.family_code for m in t2 if m is not q and m.family_code}
        grp2 = {m.group_code for m in t2 if m is not q and m.group_code}
        if p.family_code and p.family_code in fam2:
            return False
        if p.group_code and p.group_code in grp2:
            return False

        return True

    if not teams:
        return teams

    while time.time() - start < timeout_seconds:
        spread_before = evaluate_spread(teams)
        best_swap = None
        best_improvement = 0.0

        for i, t1 in enumerate(teams):
            for j in range(i + 1, len(teams)):
                t2 = teams[j]
                singles1 = [p for p in t1 if not p.friend_code]
                singles2 = [q for q in t2 if not q.friend_code]
                if not singles1 or not singles2:
                    continue
                for p in singles1:
                    for q in singles2:
                        if not can_swap(p, q, t1, t2):
                            continue

                        t1_total = sum(x.average for x in t1) - p.average + q.average
                        t2_total = sum(x.average for x in t2) - q.average + p.average

                        new_totals = []
                        for k, tm in enumerate(teams):
                            if k == i:
                                new_totals.append(t1_total)
                            elif k == j:
                                new_totals.append(t2_total)
                            else:
                                new_totals.append(sum(m.average for m in tm))

                        new_spread = max(new_totals) - min(new_totals)
                        improvement = spread_before - new_spread
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_swap = (i, j, p, q)

        if not best_swap or best_improvement <= 0:
            logger.debug("No improving swap found or no beneficial improvement (time=%.2fs).", time.time() - start)
            break

        i, j, p, q = best_swap
        teams[i].remove(p)
        teams[j].remove(q)
        teams[i].append(q)
        teams[j].append(p)
        logger.debug("Applied swap between team %d and %d -> improvement %.2f", i, j, best_improvement)

    if time.time() - start >= timeout_seconds:
        logger.debug("optimize_balance_with_timeout reached timeout (%.2fs)", time.time() - start)
    else:
        logger.debug("optimize_balance_with_timeout finished in %.2fs", time.time() - start)

    return teams


# ---------------------------
# Exact search: CP-SAT model that minimizes spread, then group overlaps, under the hard constraints
# (friend blocks together, family separation, exact layout sizes).
# Averages are scaled to integers at 0.01 precision.
# ---------------------------
SPREAD_SCALE = 100


def count_group_overlaps(teams: List[List[Player]]) -> int:
    """Number of players sharing a group_code with an earlier teammate (0 = every team has distinct groups)."""
    overlaps = 0
    for team in teams:
        counts: Dict[str, int] = defaultdict(int)
        for p in team:
            if p.group_code:
                counts[p.group_code] += 1
        overlaps += sum(c - 1 for c in counts.values() if c > 1)
    return overlaps


def solve_min_spread(
    players: List[Player],
    sizes: List[int],
    spread_limit: Optional[float] = None,
    timeout_seconds: float = 10.0,
    max_group_per_team: Optional[int] = None,
) -> Tuple[List[List[Player]], float, int, bool]:
    """
    Find the minimum-spread assignment for a layout (optionally restricted to spread <= spread_limit),
    then, among assignments with that spread, the one with the fewest group overlaps.

    Returns (teams, spread, overlaps, proven):
        teams    -- best assignment found ([] if none)
        spread   -- evaluate_spread(teams), or math.inf if none
        overlaps -- count_group_overlaps(teams)
        proven   -- True if the search was exhausted (teams is optimal, or no assignment exists)
    """
    k = len(sizes)
    blocks = build_blocks(players, max(sizes))
    if any(len(b) > max(sizes) for b in blocks):
        return [], math.inf, 0, True

    model = cp_model.CpModel()
    x = [[model.NewBoolVar(f"b{b}_t{t}") for t in range(k)] for b in range(len(blocks))]

    for b in range(len(blocks)):
        model.AddExactlyOne(x[b])
    for t in range(k):
        model.Add(sum(len(blocks[b]) * x[b][t] for b in range(len(blocks))) == sizes[t])

    # Family limit is per block (as in violates_family); group counts every player.
    family_blocks: Dict[str, set] = defaultdict(set)
    group_blocks: Dict[str, List[int]] = defaultdict(list)
    for b, block in enumerate(blocks):
        for p in block:
            if p.family_code:
                family_blocks[p.family_code].add(b)
            if p.group_code:
                group_blocks[p.group_code].append(b)

    overlap_vars = []
    for t in range(k):
        for bs in family_blocks.values():
            model.Add(sum(x[b][t] for b in bs) <= 1)
        for g, bs in group_blocks.items():
            count = sum(x[b][t] for b in bs)
            if max_group_per_team is not None:
                model.Add(count <= max_group_per_team)
            overlap = model.NewIntVar(0, len(bs), f"overlap_{g}_t{t}")
            model.Add(overlap >= count - 1)
            overlap_vars.append(overlap)

    block_scores = [round(sum(p.average for p in block) * SPREAD_SCALE) for block in blocks]
    max_total = sum(abs(v) for v in block_scores)
    totals = []
    for t in range(k):
        total = model.NewIntVar(-max_total, max_total, f"total{t}")
        model.Add(total == sum(block_scores[b] * x[b][t] for b in range(len(blocks))))
        totals.append(total)

    # Teams of equal size are interchangeable; order them to skip mirrored solutions.
    for t in range(k - 1):
        if sizes[t] == sizes[t + 1]:
            model.Add(totals[t] <= totals[t + 1])

    hi = model.NewIntVar(-max_total, max_total, "hi")
    lo = model.NewIntVar(-max_total, max_total, "lo")
    model.AddMaxEquality(hi, totals)
    model.AddMinEquality(lo, totals)
    if spread_limit is not None:
        model.Add(hi - lo <= round(spread_limit * SPREAD_SCALE))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = timeout_seconds
    solver.parameters.num_workers = 1

    # Stage 1: lowest spread.
    model.Minimize(hi - lo)
    status = solver.Solve(model)
    logger.debug("CP-SAT layout=%s limit=%s spread stage -> %s (%.2fs)", sizes, spread_limit, solver.StatusName(status), solver.WallTime())
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], math.inf, 0, status == cp_model.INFEASIBLE
    proven = status == cp_model.OPTIMAL

    # Stage 2: fewest group overlaps at that spread.
    model.Add(hi - lo <= solver.Value(hi - lo))
    for row in x:
        for var in row:
            model.AddHint(var, solver.Value(var))
    model.Minimize(sum(overlap_vars))
    status = solver.Solve(model)
    logger.debug("CP-SAT layout=%s limit=%s overlap stage -> %s (%.2fs)", sizes, spread_limit, solver.StatusName(status), solver.WallTime())
    proven = proven and status == cp_model.OPTIMAL

    teams = [
        [p for b in range(len(blocks)) if solver.Value(x[b][t]) for p in blocks[b]]
        for t in range(k)
    ]
    return teams, evaluate_spread(teams), count_group_overlaps(teams), proven


# ---------------------------
# Master: spread tiers (lowest first, each exhausted before the next).
# Lower spread always wins; group overlaps are only minimized among equal-spread arrangements.
# ---------------------------
def balance_teams(players: List[Player], max_search_spread: int = 50, solver_timeout_seconds: float = 10.0) -> Dict:
    """
    Attempts to create the most balanced teams possible.

    For each spread tier (0, 10, ..., max_search_spread), every layout is searched exhaustively for the
    lowest spread within the tier, then the fewest group overlaps at that spread. A higher tier is only
    tried once the lower tier is proven impossible. Family separation, friend blocks and team sizes are
    hard constraints; group separation is not.

    Returns:
        {
            "spread_used": float | None,
            "teams": List[List[Player]] | [],
            "group_mode": "STRICT" (no group overlaps) | "RELAXED" | "NONE",
            "group_overlaps": int | None
        }
    """
    n = len(players)
    layouts = compute_team_sizes(n)
    if not layouts:
        logger.error("No valid team-size layouts for %d players.", n)
        return {"spread_used": None, "teams": [], "group_mode": "NONE", "group_overlaps": None}

    def result(teams: List[List[Player]], spread: float, overlaps: int) -> Dict:
        mode = "STRICT" if overlaps == 0 else "RELAXED"
        return {"spread_used": spread, "teams": teams, "group_mode": mode, "group_overlaps": overlaps}

    def search(spread_limit: Optional[float]) -> Tuple[Optional[List[List[Player]]], float, int, bool]:
        best = (None, math.inf, 0)
        exhausted = True
        for layout in layouts:
            teams, spread, overlaps, proven = solve_min_spread(players, layout, spread_limit, solver_timeout_seconds)
            exhausted = exhausted and proven
            logger.debug("[SPREAD=%s] Layout %s -> spread=%.2f overlaps=%d (proven=%s)", spread_limit, layout, spread, overlaps, proven)
            if teams and (spread, overlaps) < best[1:]:
                best = (teams, spread, overlaps)
        return best[0], best[1], best[2], exhausted

    for spread_limit in range(0, max_search_spread + 1, 10):
        teams, spread, overlaps, exhausted = search(spread_limit)
        if teams is not None:
            if exhausted:
                logger.info("[SPREAD=%d] Success (spread=%.2f, group overlaps=%d, proven minimum)", spread_limit, spread, overlaps)
            else:
                logger.warning("[SPREAD=%d] Success (spread=%.2f, group overlaps=%d, solver timed out before proving minimum)", spread_limit, spread, overlaps)
            return result(teams, spread, overlaps)

        if exhausted:
            logger.info("[SPREAD=%d] No arrangement possible (all options exhausted)", spread_limit)
        else:
            logger.warning("[SPREAD=%d] Solver timed out before exhausting this tier; moving on", spread_limit)

    # Best-effort: lowest spread with no limit, heuristic if the solver finds nothing in time
    logger.info("No arrangement within spread=%d. Attempting final best-effort search across layouts.", max_search_spread)
    teams, spread, overlaps, _ = search(None)
    if teams is None:
        for layout in layouts:
            assigned = try_assign_with_timeout(players, layout, max_group_per_team=2, timeout_seconds=5.0)
            if not assigned:
                continue
            optimized = optimize_balance_with_timeout(assigned, timeout_seconds=5.0)
            if evaluate_spread(optimized) < spread:
                teams, spread, overlaps = optimized, evaluate_spread(optimized), count_group_overlaps(optimized)

    if teams is not None:
        logger.warning("No arrangement met spread<=%d; returning best-effort with spread=%.2f", max_search_spread, spread)
        return result(teams, spread, overlaps)

    logger.error("No arrangement found (even best-effort failed).")
    return {"spread_used": None, "teams": [], "group_mode": "NONE", "group_overlaps": None}
