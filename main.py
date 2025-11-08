from utils.csv_handler import read_players_from_csv
from utils.json_handler import write_teams_to_json
from balancer.team_balancer import balance_teams


def main():
    input_csv = "data/players.csv"
    output_json = "data/teams.json"

    players = read_players_from_csv(input_csv)
    result = balance_teams(players)

    if not result["teams"]:
        print("No valid team arrangement found.")
        return

    spread_used = result["spread_used"]
    teams = result["teams"]

    print(f"\n=== Final Spread Used: {spread_used} ===\n")

    # Summary output
    for i, team in enumerate(teams, 1):
        total = sum(p.average for p in team)
        avg = total / len(team)
        groups = [p.group_code for p in team]
        print(
            f"Team {i}: {len(team)} players | Total={total:.2f} | Avg={avg:.2f} | Groups={groups}"
        )

    for i, team in enumerate(teams, 1):
        names = [p.name for p in team]
        print(
            f"Team {i}: {names}"
        )

    write_teams_to_json(teams, output_json, spread_used)
    print(f"\nTeams written to {output_json}")


if __name__ == "__main__":
    main()
