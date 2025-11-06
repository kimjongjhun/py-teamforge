import csv
from models.player import Player


def read_players_from_csv(filename: str):
    players = []

    with open(filename, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            players.append(
                Player(
                    name=row["name"],
                    average=float(row["average"]),
                    group_code=row["group_code"],
                    family_code=row["family_code"],
                    friend_code=row["friend_code"],
                )
            )

    return players
