# balancer/team_balancer.py
from typing import List, Dict, Optional
from models.player import Player
import math
import time
import heapq
from collections import defaultdict

# ---------------------------------------------------------
# Helpers: spread evaluation
# ---------------------------------------------------------
def evaluate_spread(teams: List[List[Player]]) -> float:
    totals = [sum(p.average for p in t) for t in teams if len(t) > 0]
    if not totals:
        return math.inf
    return max(totals) - min(totals)


# ---------------------------------------------------------
# Team-size distributions (4s, up to two 3s, or one 5)
# ---------------------------------------------------------
def compute_team_sizes(n: int) -> List[List[int]]:
    layouts: List[List[int]] = []

    # all 4s
    if n % 4 == 0:
        layouts.append([4] * (n // 4))

    # one 5
    if n >= 5 and (n - 5) % 4 == 0:
        layouts.append([5] + [4] * ((n - 5) // 4))

    # one 3
    if n >= 3 and (n - 3) % 4 == 0:
        layouts.append([3] + [4] * ((n - 3) // 4))

    # two 3s
    if n >= 6 and (n - 6) % 4 == 0:
        layouts.append([3, 3] + [4] * ((n - 6) // 4))

    return layouts


# ---------------------------------------------------------
# Build friend blocks; split only when block > max_capacity
# ---------------------------------------------------------
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

    for grp in friend_map.values():
        # Sort group internally by descending average (useful when splitting)
        sorted_grp = sorted(grp, key=lambda x: -x.average)
        if len(sorted_grp) <= max_capacity:
            blocks.append(sorted_grp)
        else:
            # split into chunks sized max_capacity
            while sorted_grp:
                chunk = sorted_grp[:max_capacity]
                blocks.append(chunk)
                sorted_grp = sorted_grp[max_capacity:]

    # add singles as single-element blocks
    for s in singles:
        blocks.append([s])

    # sort blocks: larger blocks first (reduces backtracking), tie-break by block avg desc
    blocks.sort(key=lambda b: (-len(b), -sum(p.average for p in b)))
    return blocks


# ---------------------------------------------------------
# Constraint checks: family and group (group codes must be split)
# ---------------------------------------------------------
def violates_family(team: List[Player], block: List[Player]) -> bool:
    existing = {p.family_code for p in team if p.family_code}
    for p in block:
        if p.family_code and p.family_code in existing:
            return True
    return False


def violates_group(team: List[Player], block: List[Player]) -> bool:
    existing = {p.group_code for p in team if p.group_code}
    for p in block:
        if p.group_code and p.group_code in existing:
            return True
    return False


# ---------------------------------------------------------
# Optimistic spread estimator used for pruning during backtracking.
# Greedily assigns remaining players (flattened) to lowest-total eligible teams.
# ---------------------------------------------------------
def optimistic_spread_if_fill(teams: List[List[Player]], remaining_blocks: List[List[Player]], sizes: List[int]) -> float:
    # current totals and capacities
    totals = [sum(p.average for p in t) for t in teams]
    caps = [sizes[i] - len(teams[i]) for i in range(len(sizes))]

    # flatten remaining players averages
    remaining_avgs: List[float] = []
    for blk in remaining_blocks:
        for p in blk:
            remaining_avgs.append(p.average)
    remaining_avgs.sort(reverse=True)  # place largest first

    # min-heap of (current_total, team_index) only for teams with capacity >0
    heap: List[tuple] = []
    for i, cap in enumerate(caps):
        if cap > 0:
            heap.append((totals[i], i))
    heapq.heapify(heap)

    # simulate assignment of each avg to the team with smallest total that still has capacity
    caps_copy = caps[:]
    totals_copy = totals[:]
    for avg in remaining_avgs:
        # find a team with capacity
        if not heap:
            # no capacity left, return current spread
            break
        total, ti = heapq.heappop(heap)
        totals_copy[ti] += avg
        caps_copy[ti] -= 1
        if caps_copy[ti] > 0:
            heapq.heappush(heap, (totals_copy[ti], ti))

    if not totals_copy:
        return math.inf
    return max(totals_copy) - min(totals_copy)


# ---------------------------------------------------------
# Backtracking assignment with optimistic pruning and timeout
# ---------------------------------------------------------
def try_assign_with_timeout(players: List[Player], sizes: List[int], timeout_seconds: float = 10.0) -> List[List[Player]]:
    start_time = time.time()
    max_capacity = max(sizes)
    blocks = build_blocks(players, max_capacity)

    # Quick infeasibility check: any block larger than largest size -> impossible
    if any(len(b) > max_capacity for b in blocks):
        return []

    teams: List[List[Player]] = [[] for _ in sizes]
    n_blocks = len(blocks)

    # precompute players remaining counts for quick capacity prune
    remaining_players_from = [0] * (n_blocks + 1)
    for i in range(n_blocks - 1, -1, -1):
        remaining_players_from[i] = remaining_players_from[i + 1] + len(blocks[i])

    def backtrack(idx: int) -> bool:
        # timeout check
        if time.time() - start_time > timeout_seconds:
            # timed out
            # print("  [!] try_assign timed out")
            return False

        # capacity feasibility prune
        remaining = remaining_players_from[idx]
        free_slots = sum(sizes[i] - len(teams[i]) for i in range(len(sizes)))
        if remaining > free_slots:
            return False

        if idx == n_blocks:
            # all placed
            return True

        block = blocks[idx]

        # heuristic: try teams in order of increasing total (attempt to keep balanced)
        order = sorted(range(len(teams)), key=lambda i: sum(p.average for p in teams[i]))

        for ti in order:
            # size check
            if len(teams[ti]) + len(block) > sizes[ti]:
                continue
            # family and group constraints
            if violates_family(teams[ti], block) or violates_group(teams[ti], block):
                continue

            # tentatively place block
            teams[ti].extend(block)

            # optimistic pruning: compute minimal achievable spread if we greedily fill remaining blocks
            optimistic = optimistic_spread_if_fill(teams, blocks[idx + 1:], sizes)
            # if optimistic is inf (no teams), we still allow continuing to check capacity
            # prune if even optimistic spread cannot meet current spread target — but we don't know that here;
            # we simply prune nothing based on spread here because spread threshold is handled in progressive loop.
            # However we still use optimistic to skip obviously bad states when it is huge
            # We will not prune aggressively here, only use optimistic to detect impossibility when it's absurdly large.
            if optimistic < math.inf:
                # continue with backtracking
                if backtrack(idx + 1):
                    return True
            else:
                # undo and try other team
                pass

            # undo placement
            for _ in block:
                teams[ti].pop()

        return False

    ok = backtrack(0)
    return teams if ok else []


# ---------------------------------------------------------
# Optimization pass: time-limited 1-for-1 singleton swaps.
# Does not break friend groups (only swaps players without friend_code).
# ---------------------------------------------------------
def optimize_balance_with_timeout(teams: List[List[Player]], timeout_seconds: float = 10.0) -> List[List[Player]]:
    start = time.time()

    def can_swap(p: Player, q: Player, t1: List[Player], t2: List[Player]) -> bool:
        # do not break friend groups: only allow swap if both players have no friend_code
        if p.friend_code or q.friend_code:
            return False

        # after swap, families and group constraints must hold
        # t1 after swap: remove p, add q
        fam1 = {m.family_code for m in t1 if m is not p and m.family_code}
        grp1 = {m.group_code for m in t1 if m is not p and m.group_code}
        if q.family_code and q.family_code in fam1:
            return False
        if q.group_code and q.group_code in grp1:
            return False

        # t2 after swap: remove q, add p
        fam2 = {m.family_code for m in t2 if m is not q and m.family_code}
        grp2 = {m.group_code for m in t2 if m is not q and m.group_code}
        if p.family_code and p.family_code in fam2:
            return False
        if p.group_code and p.group_code in grp2:
            return False

        return True

    if not teams:
        return teams

    # repeatedly try to find a swap that improves spread until timeout
    while time.time() - start < timeout_seconds:
        spread_before = evaluate_spread(teams)
        best_swap = None
        best_improvement = 0.0

        # iterate pairs of teams
        for i, t1 in enumerate(teams):
            for j in range(i + 1, len(teams)):
                t2 = teams[j]
                # consider only singleton candidates to avoid breaking friend groups
                singles1 = [p for p in t1 if not p.friend_code]
                singles2 = [q for q in t2 if not q.friend_code]
                if not singles1 or not singles2:
                    continue
                for p in singles1:
                    for q in singles2:
                        if not can_swap(p, q, t1, t2):
                            continue

                        # compute new totals after hypothetical swap
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
            break  # no improving swap found or timeout

        # apply best swap
        i, j, p, q = best_swap
        teams[i].remove(p)
        teams[j].remove(q)
        teams[i].append(q)
        teams[j].append(p)

    return teams


# ---------------------------------------------------------
# Master function: progressive relaxation with logging and 10s timeouts
# ---------------------------------------------------------
def balance_teams(players: List[Player], max_search_spread: int = 50) -> Dict:
    """
    Attempts to create the most balanced teams possible.
    Start with spread=0, increasing by +10 until a valid arrangement is found or max_search_spread reached.

    Returns:
      {
        "spread_used": float | None,
        "teams": List[List[Player]] | []
      }
    """
    n = len(players)
    layouts = compute_team_sizes(n)
    if not layouts:
        return {"spread_used": None, "teams": []}

    # try spreads from 0 to max_search_spread in steps of 10
    for spread_limit in range(0, max_search_spread + 1, 10):
        print(f"Trying spread: {spread_limit}")
        best_teams_for_limit: Optional[List[List[Player]]] = None
        best_spread_for_limit = math.inf

        # try each layout
        for layout in layouts:
            # attempt assign with timeout
            assigned = try_assign_with_timeout(players, layout, timeout_seconds=10.0)
            if not assigned:
                # timed out or impossible for this layout; skip
                continue

            # optimize with timeout
            optimized = optimize_balance_with_timeout(assigned, timeout_seconds=10.0)

            spread = evaluate_spread(optimized)

            # if spread satisfies current limit, return immediately (we prefer first valid)
            if spread <= spread_limit:
                return {"spread_used": spread, "teams": optimized}

            # otherwise, track the best attempt (lowest spread) under this limit pass
            if spread < best_spread_for_limit:
                best_spread_for_limit = spread
                best_teams_for_limit = optimized

        # no exact fit for this spread_limit; continue to next (larger) spread
        # loop continues, trying next spread_limit

    # after all spreads up to max_search_spread tried, if nothing satisfied
    # return the best attempt found across all layouts at final step if exists, otherwise failure
    # To keep behavior deterministic, we will attempt a final pass to find the best spread achieved (if any)
    # Try all layouts once more without spread restriction but with timeouts, and choose the best spread seen
    print("No arrangement met the spread limits; searching for best-effort arrangement (time-limited).")
    best_overall = None
    best_overall_spread = math.inf
    for layout in layouts:
        assigned = try_assign_with_timeout(players, layout, timeout_seconds=5.0)
        if not assigned:
            continue
        optimized = optimize_balance_with_timeout(assigned, timeout_seconds=5.0)
        spread = evaluate_spread(optimized)
        if spread < best_overall_spread:
            best_overall_spread = spread
            best_overall = optimized

    if best_overall is not None:
        return {"spread_used": best_overall_spread, "teams": best_overall}

    return {"spread_used": None, "teams": []}
