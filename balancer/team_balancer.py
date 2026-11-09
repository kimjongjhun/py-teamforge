# balancer/team_balancer.py
import math
from typing import Dict, List, Optional, Tuple

from balancer.cpsat_solver import find_min_group_overlaps, find_min_spread_assignment
from balancer.heuristic_search import assign_blocks_with_backtracking, improve_balance_with_swaps
from balancer.logging_config import logger
from balancer.scoring import calculate_spread, count_group_overlaps
from balancer.team_layouts import get_valid_team_size_layouts
from models.player import Player


def build_result(teams: List[List[Player]], spread: float, group_overlaps: int) -> Dict:
    group_mode = "STRICT" if group_overlaps == 0 else "RELAXED"
    return {"spread_used": spread, "teams": teams, "group_mode": group_mode, "group_overlaps": group_overlaps}


def balance_teams(players: List[Player], max_search_spread: int = 50, solver_timeout_seconds: float = 10.0) -> Dict:
    """
    Attempts to create the most balanced teams possible.

    Group members share a team only when mathematically required: each group is capped at
    ceil(group size / teams) per team and the number of overlaps is the minimum the hard constraints
    (sizes, friend blocks, families) allow. Within that, each spread tier (0, 10, ..., max_search_spread)
    is searched exhaustively for the lowest spread, and a higher tier is only tried once the lower tier
    is proven impossible.

    Returns:
        {
            "spread_used": float | None,
            "teams": List[List[Player]] | [],
            "group_mode": "STRICT" (no group overlaps) | "RELAXED" | "NONE",
            "group_overlaps": int | None
        }
    """
    no_result = {"spread_used": None, "teams": [], "group_mode": "NONE", "group_overlaps": None}
    layouts = get_valid_team_size_layouts(len(players))
    if not layouts:
        logger.error("No valid team-size layouts for %d players.", len(players))
        return no_result

    enforce_group_caps = True
    min_overlaps_by_layout: Dict[Tuple[int, ...], int] = {}
    for enforce_group_caps in (True, False):
        for layout in layouts:
            min_overlaps, proven = find_min_group_overlaps(players, layout, solver_timeout_seconds, enforce_group_caps)
            if not proven:
                logger.warning("Solver timed out proving minimum group overlaps for layout %s", layout)
            if min_overlaps is not None:
                min_overlaps_by_layout[tuple(layout)] = min_overlaps
        if min_overlaps_by_layout:
            break
        logger.warning("No assignment fits the per-team group caps; retrying without them.")

    best_layouts: List[List[int]] = []
    required_overlaps: Optional[int] = None
    if min_overlaps_by_layout:
        required_overlaps = min(min_overlaps_by_layout.values())
        best_layouts = [list(layout) for layout, overlaps in min_overlaps_by_layout.items() if overlaps == required_overlaps]
        logger.info("Minimum group overlaps required: %d (layouts %s)", required_overlaps, best_layouts)

    def search_layouts(spread_limit: Optional[float]) -> Tuple[Optional[List[List[Player]]], float, int, bool]:
        best_teams, best_spread, best_overlaps = None, math.inf, 0
        all_proven = True
        for layout in best_layouts:
            teams, spread, overlaps, proven = find_min_spread_assignment(
                players, layout, spread_limit, solver_timeout_seconds, required_overlaps, enforce_group_caps
            )
            all_proven = all_proven and proven
            logger.debug("[SPREAD=%s] Layout %s -> spread=%.2f overlaps=%d (proven=%s)", spread_limit, layout, spread, overlaps, proven)
            if teams and (spread, overlaps) < (best_spread, best_overlaps):
                best_teams, best_spread, best_overlaps = teams, spread, overlaps
        return best_teams, best_spread, best_overlaps, all_proven

    if best_layouts:
        for spread_limit in range(0, max_search_spread + 1, 10):
            teams, spread, overlaps, all_proven = search_layouts(spread_limit)
            if teams is not None:
                if all_proven:
                    logger.info("[SPREAD=%d] Success (spread=%.2f, group overlaps=%d, proven minimum)", spread_limit, spread, overlaps)
                else:
                    logger.warning("[SPREAD=%d] Success (spread=%.2f, group overlaps=%d, solver timed out before proving minimum)", spread_limit, spread, overlaps)
                return build_result(teams, spread, overlaps)

            if all_proven:
                logger.info("[SPREAD=%d] No arrangement possible (all options exhausted)", spread_limit)
            else:
                logger.warning("[SPREAD=%d] Solver timed out before exhausting this tier; moving on", spread_limit)

        logger.info("No arrangement within spread=%d. Attempting final best-effort search.", max_search_spread)
        teams, spread, overlaps, _ = search_layouts(None)
        if teams is not None:
            logger.warning("No arrangement met spread<=%d; returning best-effort with spread=%.2f", max_search_spread, spread)
            return build_result(teams, spread, overlaps)

    # Last resort if the solver found nothing in time: original heuristic
    best_teams, best_spread = None, math.inf
    for layout in layouts:
        assigned = assign_blocks_with_backtracking(players, layout, max_group_per_team=2, timeout_seconds=5.0)
        if not assigned:
            continue
        improved = improve_balance_with_swaps(assigned, timeout_seconds=5.0)
        if calculate_spread(improved) < best_spread:
            best_teams, best_spread = improved, calculate_spread(improved)

    if best_teams is not None:
        logger.warning("Solver found nothing; returning heuristic best-effort with spread=%.2f", best_spread)
        return build_result(best_teams, best_spread, count_group_overlaps(best_teams))

    logger.error("No arrangement found (even best-effort failed).")
    return no_result
