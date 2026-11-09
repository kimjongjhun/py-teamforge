# balancer/team_layouts.py
from typing import List

from balancer.logging_config import logger


def get_valid_team_size_layouts(player_count: int) -> List[List[int]]:
    """Allowed team-size layouts: all 4s, one 5 + 4s, one 3 + 4s, or two 3s + 4s."""
    layouts: List[List[int]] = []

    if player_count % 4 == 0:
        layouts.append([4] * (player_count // 4))

    if player_count >= 5 and (player_count - 5) % 4 == 0:
        layouts.append([5] + [4] * ((player_count - 5) // 4))

    if player_count >= 3 and (player_count - 3) % 4 == 0:
        layouts.append([3] + [4] * ((player_count - 3) // 4))

    if player_count >= 6 and (player_count - 6) % 4 == 0:
        layouts.append([3, 3] + [4] * ((player_count - 6) // 4))

    logger.debug("Computed layouts for %d players: %s", player_count, layouts)
    return layouts
