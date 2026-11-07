# Scenario Test Schema (Contract C-SCN)

Scenarios specify self-contained, reproducible rule demonstrations that are shared directly between pytest regression suites (`tests/test_scenarios.py`) and the browser application (`web/backend/scenarios.py` and the Rules page).

## JSON Structure

```json
{
  "name": "scenario_identifier",
  "title": "Short Human-Readable Title",
  "description": "Markdown description displayed in the web demo explaining what rule mechanic is demonstrated.",
  "config": {
    "board_size": 5,
    "A": false,
    "B": false,
    "C": true,
    "K": 3,
    "M": 0
  },
  "start": "initial",
  "script": [
    ["BLOCK", [3, 2]],
    ["MOVE", [4, 1]]
  ],
  "expect": [
    {
      "after_ply": 1,
      "blocked": {"3,2": 3},
      "legal_excludes": [["MOVE", [3, 2]]],
      "note": "Timer starts at K"
    },
    {
      "after_ply": 4,
      "blocked": {},
      "note": "Expired"
    },
    {
      "after_ply": 0,
      "illegal": ["WALL", [0, 0, "H"]],
      "error_contains": "no walls remaining"
    }
  ],
  "final": {
    "terminal_reason": null,
    "winner": null
  }
}
```

## Field Specification

- `name` (string): Unique identifier for the scenario.
- `title` (string): Title for display.
- `description` (string): Markdown formatted text explaining the scenario and rules verified.
- `config` (object): Engine configuration parameters (`board_size`, `A`, `B`, `C`, `K`, `M`, optional `max_plies`).
- `start`: Either `"initial"` (fresh engine) or an object `{"raw_key": [...]}` to restore an arbitrary engine state.
- `script` (array of actions): Sequential actions applied with `update_clock=False`.
- `expect` (array of expectation objects):
  - `after_ply` (int): Checked after applying `script[:after_ply]`. `after_ply: 0` checks before any script action is applied.
  - `blocked` (object): Dict mapping `"r,c"` coordinate strings to remaining integer timer values.
  - `inventory` (array of 2 ints): `[inv_p0, inv_p1]`.
  - `hop` (array of 2 bools): `[hop_p0, hop_p1]`.
  - `block` (array of 2 bools): `[block_p0, block_p1]`.
  - `latest` (array of 2 wall triples or nulls): `[latest_p0, latest_p1]`.
  - `to_move` (int): Current player to move (0 or 1).
  - `legal_includes` (array of actions): Actions that must be in `engine.get_legal_actions()`.
  - `legal_excludes` (array of actions): Actions that must NOT be in `engine.get_legal_actions()`.
  - `illegal` (action): Action that must raise `ValueError` when applied on an engine clone.
  - `error_contains` (string): Substring that must appear in the raised `ValueError`.
  - `terminal_reason` (string or null): Expected `engine.terminal_reason`.
  - `winner` (int or null): Expected `engine.winner`.
- `final` (object): Expectations evaluated after the entire script has finished.
