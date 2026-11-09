# balancer/cpsat_solver.py
# Exact search with OR-Tools CP-SAT.
# Hard constraints: exact layout sizes, friend blocks together, family separation, and the per-team
# group caps from calculate_group_caps. Averages are scaled to integers at 0.01 precision.
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from ortools.sat.python import cp_model

from balancer.constraints import calculate_group_caps
from balancer.friend_blocks import build_friend_blocks
from balancer.logging_config import logger
from balancer.scoring import calculate_spread, count_group_overlaps
from models.player import Player

AVERAGE_SCALE = 100


class TeamAssignmentModel:
    """CP-SAT model with one boolean per (friend block, team): is this block on this team?"""

    def __init__(self, players: List[Player], team_sizes: List[int], enforce_group_caps: bool):
        self.team_count = len(team_sizes)
        self.blocks = build_friend_blocks(players, max(team_sizes))
        self.all_blocks_fit = all(len(block) <= max(team_sizes) for block in self.blocks)
        self.model = cp_model.CpModel()

        block_indices = range(len(self.blocks))
        team_indices = range(self.team_count)
        self.block_on_team = [
            [self.model.NewBoolVar(f"block{b}_on_team{t}") for t in team_indices] for b in block_indices
        ]

        for b in block_indices:
            self.model.AddExactlyOne(self.block_on_team[b])
        for t in team_indices:
            self.model.Add(sum(len(self.blocks[b]) * self.block_on_team[b][t] for b in block_indices) == team_sizes[t])

        # Family limit is per block (as in has_family_conflict); group counts every player.
        blocks_by_family: Dict[str, Set[int]] = defaultdict(set)
        blocks_by_group: Dict[str, List[int]] = defaultdict(list)
        for b, block in enumerate(self.blocks):
            for player in block:
                if player.family_code:
                    blocks_by_family[player.family_code].add(b)
                if player.group_code:
                    blocks_by_group[player.group_code].append(b)
        group_caps = calculate_group_caps(self.blocks, self.team_count)

        overlap_vars = []
        for t in team_indices:
            for family_block_indices in blocks_by_family.values():
                self.model.Add(sum(self.block_on_team[b][t] for b in family_block_indices) <= 1)
            for group_code, group_block_indices in blocks_by_group.items():
                group_players_on_team = sum(self.block_on_team[b][t] for b in group_block_indices)
                if enforce_group_caps:
                    self.model.Add(group_players_on_team <= group_caps[group_code])
                overlaps_on_team = self.model.NewIntVar(0, len(group_block_indices), f"overlaps_{group_code}_team{t}")
                self.model.Add(overlaps_on_team >= group_players_on_team - 1)
                overlap_vars.append(overlaps_on_team)
        self.total_group_overlaps = sum(overlap_vars)

        scaled_block_totals = [round(sum(p.average for p in block) * AVERAGE_SCALE) for block in self.blocks]
        total_bound = sum(abs(v) for v in scaled_block_totals)
        team_totals = []
        for t in team_indices:
            team_total = self.model.NewIntVar(-total_bound, total_bound, f"team{t}_total")
            self.model.Add(team_total == sum(scaled_block_totals[b] * self.block_on_team[b][t] for b in block_indices))
            team_totals.append(team_total)

        # Teams of equal size are interchangeable; order them to skip mirrored solutions.
        for t in range(self.team_count - 1):
            if team_sizes[t] == team_sizes[t + 1]:
                self.model.Add(team_totals[t] <= team_totals[t + 1])

        highest_team_total = self.model.NewIntVar(-total_bound, total_bound, "highest_team_total")
        lowest_team_total = self.model.NewIntVar(-total_bound, total_bound, "lowest_team_total")
        self.model.AddMaxEquality(highest_team_total, team_totals)
        self.model.AddMinEquality(lowest_team_total, team_totals)
        self.scaled_spread = highest_team_total - lowest_team_total

    def solve(self, timeout_seconds: float) -> Tuple[cp_model.CpSolver, int]:
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = timeout_seconds
        solver.parameters.num_workers = 1
        return solver, solver.Solve(self.model)

    def extract_teams(self, solver: cp_model.CpSolver) -> List[List[Player]]:
        return [
            [p for b, block in enumerate(self.blocks) if solver.Value(self.block_on_team[b][t]) for p in block]
            for t in range(self.team_count)
        ]


def find_min_group_overlaps(
    players: List[Player], team_sizes: List[int], timeout_seconds: float = 10.0, enforce_group_caps: bool = True
) -> Tuple[Optional[int], bool]:
    """Fewest group overlaps any valid assignment for this layout can have. Returns (overlaps | None, proven)."""
    team_model = TeamAssignmentModel(players, team_sizes, enforce_group_caps)
    if not team_model.all_blocks_fit:
        return None, True
    team_model.model.Minimize(team_model.total_group_overlaps)
    solver, status = team_model.solve(timeout_seconds)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, status == cp_model.INFEASIBLE
    return round(solver.ObjectiveValue()), status == cp_model.OPTIMAL


def find_min_spread_assignment(
    players: List[Player],
    team_sizes: List[int],
    spread_limit: Optional[float] = None,
    timeout_seconds: float = 10.0,
    max_group_overlaps: Optional[int] = None,
    enforce_group_caps: bool = True,
) -> Tuple[List[List[Player]], float, int, bool]:
    """
    Minimum-spread assignment for a layout with at most max_group_overlaps group overlaps
    (optionally restricted to spread <= spread_limit), then the fewest overlaps at that spread.

    Returns (teams, spread, group_overlaps, proven):
        teams          -- best assignment found ([] if none)
        spread         -- calculate_spread(teams), or math.inf if none
        group_overlaps -- count_group_overlaps(teams)
        proven         -- True if the search was exhausted (teams is optimal, or no assignment exists)
    """
    team_model = TeamAssignmentModel(players, team_sizes, enforce_group_caps)
    if not team_model.all_blocks_fit:
        return [], math.inf, 0, True
    if max_group_overlaps is not None:
        team_model.model.Add(team_model.total_group_overlaps <= max_group_overlaps)
    if spread_limit is not None:
        team_model.model.Add(team_model.scaled_spread <= round(spread_limit * AVERAGE_SCALE))

    team_model.model.Minimize(team_model.scaled_spread)
    solver, status = team_model.solve(timeout_seconds)
    logger.debug("CP-SAT layout=%s limit=%s spread stage -> %s (%.2fs)", team_sizes, spread_limit, solver.StatusName(status), solver.WallTime())
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], math.inf, 0, status == cp_model.INFEASIBLE
    proven = status == cp_model.OPTIMAL

    team_model.model.Add(team_model.scaled_spread <= solver.Value(team_model.scaled_spread))
    for block_vars in team_model.block_on_team:
        for var in block_vars:
            team_model.model.AddHint(var, solver.Value(var))
    team_model.model.Minimize(team_model.total_group_overlaps)
    solver, status = team_model.solve(timeout_seconds)
    logger.debug("CP-SAT layout=%s limit=%s overlap stage -> %s (%.2fs)", team_sizes, spread_limit, solver.StatusName(status), solver.WallTime())
    proven = proven and status == cp_model.OPTIMAL

    teams = team_model.extract_teams(solver)
    return teams, calculate_spread(teams), count_group_overlaps(teams), proven
