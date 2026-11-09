# balancer/constraints.py
import math
from collections import defaultdict
from typing import Dict, List

from models.player import Player


def has_family_conflict(team: List[Player], block: List[Player]) -> bool:
    """True if any player in block shares a family_code with a player already on team."""
    families_on_team = {p.family_code for p in team if p.family_code}
    return any(p.family_code and p.family_code in families_on_team for p in block)


def exceeds_group_limit(team: List[Player], block: List[Player], max_per_group: int) -> bool:
    """True if adding block to team would put more than max_per_group players of a group_code on it."""
    max_per_group = max(max_per_group, 1)
    players_per_group: Dict[str, int] = {}
    for player in team:
        if player.group_code:
            players_per_group[player.group_code] = players_per_group.get(player.group_code, 0) + 1
    for player in block:
        if not player.group_code:
            continue
        count = players_per_group.get(player.group_code, 0) + 1
        if count > max_per_group:
            return True
        players_per_group[player.group_code] = count
    return False


def calculate_group_caps(blocks: List[List[Player]], team_count: int) -> Dict[str, int]:
    """
    Max players per group_code on one team: ceil(group size / team_count), raised only if a
    single friend block already holds more players of that group.
    """
    group_sizes: Dict[str, int] = defaultdict(int)
    largest_count_in_one_block: Dict[str, int] = defaultdict(int)
    for block in blocks:
        players_per_group: Dict[str, int] = defaultdict(int)
        for player in block:
            if player.group_code:
                group_sizes[player.group_code] += 1
                players_per_group[player.group_code] += 1
        for group_code, count in players_per_group.items():
            largest_count_in_one_block[group_code] = max(largest_count_in_one_block[group_code], count)
    return {
        group_code: max(math.ceil(size / team_count), largest_count_in_one_block[group_code])
        for group_code, size in group_sizes.items()
    }
