# balancer/scoring.py
import math
from collections import defaultdict
from typing import Dict, List

from models.player import Player


def calculate_spread(teams: List[List[Player]]) -> float:
    """Highest team total minus lowest team total (non-empty teams only)."""
    team_totals = [sum(p.average for p in team) for team in teams if len(team) > 0]
    if not team_totals:
        return math.inf
    return max(team_totals) - min(team_totals)


def count_group_overlaps(teams: List[List[Player]]) -> int:
    """Number of players sharing a group_code with an earlier teammate (0 = every team has distinct groups)."""
    overlaps = 0
    for team in teams:
        players_per_group: Dict[str, int] = defaultdict(int)
        for player in team:
            if player.group_code:
                players_per_group[player.group_code] += 1
        overlaps += sum(count - 1 for count in players_per_group.values() if count > 1)
    return overlaps
