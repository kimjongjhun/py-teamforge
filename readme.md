# py-TeamForge Team Balancer

py-TeamForge is a Python-based constraint solver that builds fair and balanced teams from a list of players with personal attributes such as church group, family code, friend code, and average rating.

The system is designed for real-world use cases where strict constraints must be respected—such as preventing families from being on the same team—while still balancing team skill levels.

---

## Features

- Supports the following team-size patterns:
  - All teams of **4**
  - One team of **5** with the rest **4**
  - Up to two teams of **3** with the rest **4**
- Keeps friend groups together, splitting only when necessary.
- Ensures family members are never placed on the same team.
- Distributes church groups across teams: at most ceil(group size / teams) players of a group per team (e.g. 11 young adults on 9 teams -> two teams get 2, never one team with 3), and only as many same-group teammates as the other constraints make unavoidable.
- Searches for the most balanced result by checking spreads from 0 → 50 in increments of 10. Each tier is searched exhaustively (OR-Tools CP-SAT) and proven impossible before the next tier is tried, and the lowest spread within the first feasible tier is returned.
- Produces a clean JSON output with:
  - Final spread used
  - All team rosters
  - Team totals and averages
  - Full player metadata

---

## Repository Structure

```
.
├── balancer/
│   └── team_balancer.py      # balance_teams: group overlaps, then spread tiers
│   └── cpsat_solver.py       # OR-Tools CP-SAT model and searches
│   └── constraints.py        # family / group rules and per-team group caps
│   └── friend_blocks.py      # friend groups that must share a team
│   └── team_layouts.py       # allowed team-size layouts
│   └── scoring.py            # spread and group-overlap scoring
│   └── heuristic_search.py   # greedy fallback search
│   └── logging_config.py     # console + file logging
├── data/
│   └── players.csv       # your input (not committed)
│   └── teams.json        # generated output
├── models/
│   └── player.py
├── utils/
│   └── csv_handler.py
│   └── json_handler.py
├── logs/                 # generated log file
├── main.py
├── requirements.txt
└── README.md
```

---

## Setup (one time)

Requires Python 3.8–3.13 (the versions OR-Tools 9.12 supports).

1. Clone the repository:

   ```
   git clone https://github.com/kimjongjhun/py-teamforge.git
   cd py-teamforge
   ```

2. Create and activate a virtual environment (recommended):

   Windows (PowerShell):
   ```
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

   macOS / Linux:
   ```
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```
   pip install -r requirements.txt
   ```

   This installs `ortools` (Google OR-Tools, used for its CP-SAT solver), which also pulls in `numpy`, `pandas` and `protobuf`. That's about 220 MB in total.

Activate the virtual environment again (step 2's activate command) whenever you open a new terminal.

---

## Input Format

Create a `data/` folder and put your roster in `data/players.csv`. The `data/` folder isn't committed to git.

### Columns

| Column        | Required | Description |
|---------------|----------|-------------|
| name          | yes      | Player name |
| average       | yes      | Numeric rating (decimals allowed) |
| group_code    | no       | Church group. Players with the same code are spread across teams. |
| family_code   | no       | Family identifier. Players with the same code are never on the same team. |
| friend_code   | no       | Friend group identifier. Players with the same code are kept on the same team. |

All five column headers must be present. Leave a cell empty if it doesn't apply.

### Example `players.csv`

```
name,average,group_code,family_code,friend_code
Alice,42,30-45,FAM1,FRIEND1
Bob,39,30-45,FAM2,
Charlie,55,45-60,FAM3,FRIEND1
Diana,44,60+,FAM4,
Evan,38,30-45,FAM2,
Fiona,50,45-60,FAM5,FRIEND2
George,47,60+,FAM6,FRIEND2
```

---

## Running the Application

From the repository root, with the virtual environment active:

```
python main.py
```

The system will:

1. Load players from `data/players.csv`.
2. Pick the valid team-size layout for the number of players.
3. Work out the fewest church-group overlaps that are mathematically required.
4. Test spreads 0, 10, 20, 30, 40, 50. Each tier is searched exhaustively and proven impossible before the next one is tried.
5. Return the lowest spread found in the first feasible tier, print the teams, and write them to `data/teams.json`.

Example console output for the roster above:

```
[INFO] Minimum group overlaps required: 1 (layouts [[3, 4]])
[INFO] [SPREAD=0] No arrangement possible (all options exhausted)
...
[INFO] [SPREAD=50] Success (spread=43.00, group overlaps=1, proven minimum)

=== Final Spread Used: 43.0 ===

Team 1: 3 players | Total=136.00 | Avg=45.33 | Groups=['45-60', '60+', '30-45']
Team 2: 4 players | Total=179.00 | Avg=44.75 | Groups=['45-60', '30-45', '60+', '30-45']
Team 1: ['Fiona', 'George', 'Bob']
Team 2: ['Charlie', 'Alice', 'Diana', 'Evan']

Teams written to data/teams.json
```

Spread is the difference between the highest and lowest team **totals**, so a 3-player team next to a 4-player team naturally produces a larger spread.

### Options

`balance_teams` in `main.py` accepts two optional settings:

```python
result = balance_teams(players, max_search_spread=50, solver_timeout_seconds=10.0)
```

- `max_search_spread`: the highest spread tier to search. If nothing fits, a best-effort result without a spread limit is still returned.
- `solver_timeout_seconds`: the time limit for each solver call. If it's reached, a warning is logged saying the tier wasn't fully checked.

---

## Output JSON Format

`data/teams.json` (simplified to one team):

```
{
  "spread_used": 43.0,
  "teams": [
    {
      "team_number": 1,
      "team_size": 3,
      "total_average": 136.0,
      "mean_average": 45.33,
      "players": [
        {
          "name": "Fiona",
          "average": 50.0,
          "group_code": "45-60",
          "family_code": "FAM5",
          "friend_code": "FRIEND2"
        },
        ...
      ]
    },
    ...
  ]
}
```

---

## Constraints Enforced

- Friend groups stay together if possible.
- Family members are split into separate teams.
- Church groups share a team only when mathematically required (see Features).
- Team sizes respect allowed patterns.
- Average sums must fall within the allowed spread.

Team sizes, friend groups, family separation and the church-group limits are hard constraints; spread is minimized within them.

---

## Logging

Progress is printed to the console and appended to `logs/teamforge.log`:

```
[SPREAD=0] No arrangement possible (all options exhausted)
[SPREAD=10] Success (spread=5.00, group overlaps=3, proven minimum)
```

This shows which spread tier the solver is checking and whether it was ruled out.

---

## Troubleshooting

- `ModuleNotFoundError: No module named 'ortools'`: the virtual environment isn't active, or step 3 of Setup wasn't run.
- `FileNotFoundError: ... 'data/players.csv'`: create the `data/` folder and add your CSV, then run from the repository root.
- `KeyError: 'group_code'` (or another column): the CSV header is missing that column. All five headers are required.
- `No valid team arrangement found.`: there are fewer than 3 players, or the family constraints can't be satisfied with the available teams.
