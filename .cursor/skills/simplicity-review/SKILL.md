---
name: simplicity-review
description: Use after implementing a feature, before committing, or when code feels heavy. Reviews a diff or module for over-engineering and applies behaviour-preserving simplifications. Can be invoked manually with /simplicity-review.
---

# Simplicity review

Goal: the same behaviour with less code, fewer concepts and fewer ways to break.

## How to run
1. Scope: the current diff (`git diff`) or the named module.
2. Run `pytest -q` first and record the result.
3. Work through the checklist. Collect findings as a ranked list (biggest win first): file:line, the change, why it's safe.
4. Apply only behaviour-preserving changes, one at a time, running `pytest -q` after each. If a simplification would change behaviour, propose it instead of applying it.
5. Report lines removed, concepts removed, and anything deliberately left alone.

## Checklist
- **Dead code:** unused functions, parameters, imports, branches, commented-out blocks → delete.
- **Single-use abstraction:** a class, helper or wrapper used once → inline.
- **Speculative generality:** parameters always given the same value, options with no caller, "for later" hooks → remove.
- **Class that should be a function:** `__init__` plus one method → function. Data-only class → frozen dataclass.
- **Duplication:** the same logic in two places (especially app and report) → one function in the right layer.
- **Layer leaks:** calculations in `streamlit_app.py`, file I/O in `engine/` → move (see architecture-guard).
- **Error handling:** bare `except`, `except Exception: pass`, silent fills → narrow it or let it raise.
- **Dependencies:** a package doing what pandas, numpy or the standard library already does → drop it.
- **Names:** missing units, vague names (`data`, `tmp`, `val`) → rename.
- **Comments:** restating the code → delete; a domain rule without its *why* → add one line with the source.
- **Size smells:** function > 40 lines, nesting > 3, parameters > 4 → split or simplify.

## Don't
- Trade clarity for brevity: no clever one-liners, no comprehensions nested more than two levels.
- Refactor code outside the scope; report it instead.
- Delete a test to make code simpler.
