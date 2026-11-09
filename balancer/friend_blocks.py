# balancer/friend_blocks.py
from collections import defaultdict
from typing import Dict, List

from balancer.logging_config import logger
from models.player import Player


def build_friend_blocks(players: List[Player], max_team_size: int) -> List[List[Player]]:
    """
    Group players that must share a team. Friend groups stay together unless larger than
    max_team_size, in which case they are split into chunks; players without a friend_code
    become single-player blocks. Larger blocks come first.
    """
    players_by_friend_code: Dict[str, List[Player]] = defaultdict(list)
    players_without_friends: List[Player] = []

    for player in players:
        friend_code = player.friend_code.strip() if player.friend_code else None
        if friend_code:
            players_by_friend_code[friend_code].append(player)
        else:
            players_without_friends.append(player)

    blocks: List[List[Player]] = []

    for friend_code, friends in players_by_friend_code.items():
        friends_by_average = sorted(friends, key=lambda p: -p.average)
        if len(friends_by_average) <= max_team_size:
            blocks.append(friends_by_average)
            logger.debug("Friend block kept as-is (key=%s, size=%d)", friend_code, len(friends_by_average))
        else:
            logger.debug(
                "Friend block (key=%s) larger than max_capacity (%d). Splitting into chunks.",
                friend_code,
                max_team_size,
            )
            while friends_by_average:
                blocks.append(friends_by_average[:max_team_size])
                friends_by_average = friends_by_average[max_team_size:]

    for player in players_without_friends:
        blocks.append([player])

    # Larger blocks first (reduces backtracking), then higher block total
    blocks.sort(key=lambda block: (-len(block), -sum(p.average for p in block)))
    logger.debug("Built %d blocks (max_capacity=%d). Blocks sizes: %s", len(blocks), max_team_size, [len(b) for b in blocks])
    return blocks
