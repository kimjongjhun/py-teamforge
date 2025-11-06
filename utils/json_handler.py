import json
from typing import List
from models.player import Player


def write_teams_to_json(teams: List[List[Player]], filename: str, spread_used: float):
    output = {
        "spread_used": spread_used,
        "teams": []
    }

    for i, team in enumerate(teams, start=1):
        total = sum(p.average for p in team)
        avg = total / len(team)

        team_entry = {
            "team_number": i,
            "team_size": len(team),
            "total_average": round(total, 2),
            "mean_average": round(avg, 2),
            "players": [
                {
                    "name": p.name,
                    "average": p.average,
                    "group_code": p.group_code,
                    "family_code": p.family_code,
                    "friend_code": p.friend_code,
                }
                for p in team
            ]
        }

        output["teams"].append(team_entry)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
