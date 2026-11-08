# AI use log

The Yuva Yodha T&C forbid *unauthorized* AI-generated code. We disclose every substantive use of AI here.
A human reads and understands every AI-written line before it is committed.

| date | files | what the AI did | tool | reviewed by |
|---|---|---|---|---|
| 2026-11-08 | `.cursor/rules/*`, `.cursor/skills/*` (except the two below) | Drafted project rules and agent skills | Claude (Anthropic) | pending |
| 2026-11-08 | `.cursor/skills/test-driven-development`, `.cursor/skills/verification-before-completion` | Copied from obra/superpowers (MIT); TDD skill scoped to engine/tests | Claude (Anthropic) | pending |
| 2026-11-08 | `docs/build_plan.md`, `.gitignore` | Drafted build plan and interfaces | Claude (Anthropic) | pending |
| 2026-11-08 | `requirements.txt`, `pyproject.toml`, `.gitattributes`, `saveproof/__init__.py`, `saveproof/config.py`, `saveproof/contract.py`, `tests/test_contract.py`, `docs/build_plan.md` (constant renames), `.cursor/rules/00-saveproof-context.mdc` (period type cell) | Task 0 and follow-up: wrote contract and PlantMeta tests first (red, then green), then validate(), PlantMeta checks (Timestamp fields, tz-naive, midnight) and config constants with the sources and meanings the user specified; follow-ups: empty-frame, numeric and finite checks, single production_unit; repo hygiene files | Claude (Anthropic) | pending |
