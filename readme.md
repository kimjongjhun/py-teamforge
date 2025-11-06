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
- Distributes church groups across teams (groups must be split).
- Searches for the most balanced result by checking spreads from 0 → 50 in increments of 10.
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
│   └── team_balancer.py
├── data/
│   └── players.csv
│   └── teams.json
├── models/
│   └── player.py
├── utils/
│   └── csv_handler.py
│   └── json_handler.py
├── main.py
└── README.md
```

---

## Installation

Clone the repository:

```
git clone <your-repo-url>
cd py-Teamforge
```

Install dependencies (if any):

```
pip install -r requirements.txt
```

---

## Input Format

Place a file named `players.csv` in the `data/` directory.

### Columns

| Column        | Description |
|---------------|-------------|
| name          | Player name |
| average       | Numeric rating |
| group         | Church group |
| family_code   | Family identifier |
| friend_code   | Friend group identifier |

### Example `players.csv`

```
name,average,group,family_code,friend_code
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

Run:

```
python main.py
```

The system will:

1. Load players from the CSV.
2. Attempt valid team layouts.
3. Test spreads: 0, 10, 20, 30, 40, 50.
4. Stop at the first spread that yields valid teams.
5. Output results to `data/teams.json`.

---

## Output JSON Format

### Example Output (simplified)

```
{
  "spread": 20,
  "teams": [
    {
      "team_number": 1,
      "total_average": 135,
      "players": [
        {
          "name": "Alice",
          "average": 42,
          "group": "30-45",
          "family_code": "FAM1",
          "friend_code": "FRIEND1"
        },
        {
          "name": "George",
          "average": 47,
          "group": "60+",
          "family_code": "FAM6",
          "friend_code": "FRIEND2"
        },
        {
          "name": "Evan",
          "average": 38,
          "group": "30-45",
          "family_code": "FAM2",
          "friend_code": ""
        }
      ]
    },
    {
      "team_number": 2,
      "total_average": 150,
      "players": [
        {
          "name": "Charlie",
          "average": 55,
          "group": "45-60",
          "family_code": "FAM3",
          "friend_code": "FRIEND1"
        },
        {
          "name": "Diana",
          "average": 44,
          "group": "60+",
          "family_code": "FAM4",
          "friend_code": ""
        },
        {
          "name": "Fiona",
          "average": 50,
          "group": "45-60",
          "family_code": "FAM5",
          "friend_code": "FRIEND2"
        }
      ]
    }
  ]
}
```

---

## Constraints Enforced

- Friend groups stay together if possible.
- Family members are split into separate teams.
- Church groups are distributed to maximize diversity.
- Team sizes respect allowed patterns.
- Average sums must fall within the allowed spread.

If a perfect spread is impossible, the solver gradually relaxes constraints.

---

## Logging

During execution, you will see:

```
Trying spread: 0
Trying spread: 10
Trying spread: 20
```

This indicates which spread level the solver is working on.

