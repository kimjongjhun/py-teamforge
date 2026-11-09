# balancer/heuristic_search.py
# Original greedy search, kept as a last-resort fallback when the CP-SAT solver finds nothing in time.
import heapq
import math
import time
from typing import List

from balancer.constraints import exceeds_group_limit, has_family_conflict
from balancer.friend_blocks import build_friend_blocks
from balancer.logging_config import logger
from balancer.scoring import calculate_spread
from models.player import Player


def estimate_best_case_spread(
    teams: List[List[Player]], remaining_blocks: List[List[Player]], team_sizes: List[int]
) -> float:
    """Greedily add remaining players (highest first) to the lowest-total team with room; return the spread."""
    team_totals = [sum(p.average for p in team) for team in teams]
    open_slots = [team_sizes[i] - len(teams[i]) for i in range(len(team_sizes))]

    remaining_averages = sorted((p.average for block in remaining_blocks for p in block), reverse=True)

    lowest_total_heap = [(team_totals[i], i) for i, slots in enumerate(open_slots) if slots > 0]
    heapq.heapify(lowest_total_heap)

    for average in remaining_averages:
        if not lowest_total_heap:
            break
        _, team_index = heapq.heappop(lowest_total_heap)
        team_totals[team_index] += average
        open_slots[team_index] -= 1
        if open_slots[team_index] > 0:
            heapq.heappush(lowest_total_heap, (team_totals[team_index], team_index))

    if not team_totals:
        return math.inf
    return max(team_totals) - min(team_totals)


def assign_blocks_with_backtracking(
    players: List[Player],
    team_sizes: List[int],
    max_group_per_team: int = 1,
    timeout_seconds: float = 10.0,
) -> List[List[Player]]:
    """First assignment satisfying sizes, family and group limits (lowest-total team tried first), or []."""
    start_time = time.time()
    max_team_size = max(team_sizes)
    blocks = build_friend_blocks(players, max_team_size)

    if any(len(block) > max_team_size for block in blocks):
        logger.debug("Layout impossible: a block is larger than max_capacity=%d", max_team_size)
        return []

    teams: List[List[Player]] = [[] for _ in team_sizes]
    block_count = len(blocks)

    players_left_from_block = [0] * (block_count + 1)
    for i in range(block_count - 1, -1, -1):
        players_left_from_block[i] = players_left_from_block[i + 1] + len(blocks[i])

    def place_block(block_index: int) -> bool:
        if time.time() - start_time > timeout_seconds:
            logger.debug("assign_blocks_with_backtracking timed out after %.2fs (idx=%d)", time.time() - start_time, block_index)
            return False

        players_left = players_left_from_block[block_index]
        free_slots = sum(team_sizes[i] - len(teams[i]) for i in range(len(team_sizes)))
        if players_left > free_slots:
            logger.debug("Pruned by capacity: remaining players=%d > free_slots=%d at idx=%d", players_left, free_slots, block_index)
            return False

        if block_index == block_count:
            logger.debug("All blocks placed successfully (time=%.2fs)", time.time() - start_time)
            return True

        block = blocks[block_index]
        logger.debug("Placing block %d/%d (size=%d, avg=%.2f)", block_index + 1, block_count, len(block), sum(p.average for p in block) / len(block))

        teams_by_total = sorted(range(len(teams)), key=lambda i: sum(p.average for p in teams[i]))

        for team_index in teams_by_total:
            team = teams[team_index]
            if len(team) + len(block) > team_sizes[team_index]:
                logger.debug("  team %d rejected: capacity (%d/%d) would overflow", team_index, len(team) + len(block), team_sizes[team_index])
                continue
            if has_family_conflict(team, block):
                logger.debug("  team %d rejected: family conflict", team_index)
                continue
            if exceeds_group_limit(team, block, max_group_per_team):
                logger.debug("  team %d rejected: group conflict against max_per_group=%d", team_index, max_group_per_team)
                continue

            team.extend(block)
            logger.debug("  placed on team %d (now size %d).", team_index, len(team))
            logger.debug("    optimistic spread if filled = %.2f", estimate_best_case_spread(teams, blocks[block_index + 1:], team_sizes))

            if place_block(block_index + 1):
                return True

            for _ in block:
                team.pop()
            logger.debug("  backtracked from team %d", team_index)

        return False

    return teams if place_block(0) else []


def improve_balance_with_swaps(teams: List[List[Player]], timeout_seconds: float = 10.0) -> List[List[Player]]:
    """Repeatedly apply the best spread-reducing 1-for-1 swap of players without a friend_code."""
    start_time = time.time()

    def can_swap_players(player_a: Player, player_b: Player, team_a: List[Player], team_b: List[Player]) -> bool:
        if player_a.friend_code or player_b.friend_code:
            return False

        families_a = {m.family_code for m in team_a if m is not player_a and m.family_code}
        groups_a = {m.group_code for m in team_a if m is not player_a and m.group_code}
        if player_b.family_code and player_b.family_code in families_a:
            return False
        if player_b.group_code and player_b.group_code in groups_a:
            return False

        families_b = {m.family_code for m in team_b if m is not player_b and m.family_code}
        groups_b = {m.group_code for m in team_b if m is not player_b and m.group_code}
        if player_a.family_code and player_a.family_code in families_b:
            return False
        if player_a.group_code and player_a.group_code in groups_b:
            return False

        return True

    if not teams:
        return teams

    while time.time() - start_time < timeout_seconds:
        spread_before = calculate_spread(teams)
        best_swap = None
        best_improvement = 0.0

        for index_a, team_a in enumerate(teams):
            for index_b in range(index_a + 1, len(teams)):
                team_b = teams[index_b]
                swappable_a = [p for p in team_a if not p.friend_code]
                swappable_b = [q for q in team_b if not q.friend_code]
                if not swappable_a or not swappable_b:
                    continue
                for player_a in swappable_a:
                    for player_b in swappable_b:
                        if not can_swap_players(player_a, player_b, team_a, team_b):
                            continue

                        total_a = sum(m.average for m in team_a) - player_a.average + player_b.average
                        total_b = sum(m.average for m in team_b) - player_b.average + player_a.average
                        totals_after_swap = [
                            total_a if i == index_a else total_b if i == index_b else sum(m.average for m in team)
                            for i, team in enumerate(teams)
                        ]

                        improvement = spread_before - (max(totals_after_swap) - min(totals_after_swap))
                        if improvement > best_improvement:
                            best_improvement = improvement
                            best_swap = (index_a, index_b, player_a, player_b)

        if not best_swap or best_improvement <= 0:
            logger.debug("No improving swap found or no beneficial improvement (time=%.2fs).", time.time() - start_time)
            break

        index_a, index_b, player_a, player_b = best_swap
        teams[index_a].remove(player_a)
        teams[index_b].remove(player_b)
        teams[index_a].append(player_b)
        teams[index_b].append(player_a)
        logger.debug("Applied swap between team %d and %d -> improvement %.2f", index_a, index_b, best_improvement)

    logger.debug("improve_balance_with_swaps finished in %.2fs", time.time() - start_time)
    return teams
