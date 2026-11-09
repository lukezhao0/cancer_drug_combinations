# Handoff form

Copy this file to `docs/handoffs/YYYY-MM-DD-<milestone>.md` (or update `docs/HANDOFF.md`) when you
finish a milestone or hand work to someone else. Keep answers short; link to details.

| Field | Answer |
|:--|:--|
| **Date** | |
| **Author(s)** | |
| **Milestone / task** | e.g. M2 Oracle and splits (`docs/target_design.md` §5) |
| **Branch / PR** | |
| **Status** | done / partial / blocked |

## 1. Goal of this piece of work
*One or two sentences: what it was for, and which research question it serves.*

## 2. What was done
| Item | Where (file / module) | Verified how (test, check, number) |
|:--|:--|:--|
| | | |

## 3. Decisions taken
*Continue the numbering in `docs/project-notes/m1-freeze-and-splits.md` §2.*

| # | Decision | Alternatives considered | Why | Decided by |
|:-:|:--|:--|:--|:--|
| | | | | |

## 4. Results and key numbers
| Quantity | Value | Source file |
|:--|--:|:--|
| | | |

## 5. Caveats and known problems
- *Anything that could mislead someone using the outputs.*

## 6. How to reproduce
```bash
# exact commands, in order
```
- **Data needed that isn't in git:**
- **Expected outputs / hashes:**

## 7. Tests
- Unit: `uv run pytest -q`: *result*
- Real data: `COMBO_INTEGRATION=1 uv run pytest -q -k real`: *result*

## 8. Open questions (need a team decision)
| Question | Options | Recommendation | Owner |
|:--|:--|:--|:--|
| | | | |

## 9. Next steps
1.
2.

## 10. Files added or changed
| Path | Change |
|:--|:--|
| | |
