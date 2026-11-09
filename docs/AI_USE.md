# AI use log

The Yuva Yodha T&C forbid *unauthorized* AI-generated code. We disclose every substantive use of AI here.
A human reads and understands every AI-written line before it is committed.

| date | files | what the AI did | tool | reviewed by |
|---|---|---|---|---|
| 2026-11-08 | `.cursor/rules/*`, `.cursor/skills/*` (except the two below) | Drafted project rules and agent skills | Claude (Anthropic) | pending |
| 2026-11-08 | `.cursor/skills/test-driven-development`, `.cursor/skills/verification-before-completion` | Copied from obra/superpowers (MIT); TDD skill scoped to engine/tests | Claude (Anthropic) | pending |
| 2026-11-08 | `docs/build_plan.md`, `.gitignore` | Drafted build plan and interfaces | Claude (Anthropic) | pending |
| 2026-11-08 | `requirements.txt`, `pyproject.toml`, `.gitattributes`, `saveproof/__init__.py`, `saveproof/config.py`, `saveproof/contract.py`, `tests/test_contract.py`, `docs/build_plan.md` (constant renames), `.cursor/rules/00-saveproof-context.mdc` (period type cell) | Task 0 and follow-up: wrote contract and PlantMeta tests first (red, then green), then validate(), PlantMeta checks (Timestamp fields, tz-naive, midnight) and config constants with the sources and meanings the user specified; follow-ups: empty-frame, numeric and finite checks, single production_unit; repo hygiene files | Claude (Anthropic) | VK |
| 2026-11-08 | `docs/validation_plan.md`, `.cursor/skills/validation-testing/SKILL.md`, `docs/build_plan.md` (R1 proxies, verdict pointer) | Drafted pre-registered validation plan after checking UCI data (Load_Type is a tariff clock; strong seasonal drift); aligned skill and build plan | Claude (Anthropic) | VK |
| 2026-11-09 | `saveproof/engine/__init__.py`, `saveproof/engine/stats.py`, `tests/engine/test_stats.py`, `saveproof/config.py` (3 FSU constants) | V1: wrote hand-computed 6-row tests first (red: module missing), then cvrmse, nmbe, lag1_autocorr, n_effective, savings_uncertainty_kwh; daily form ported from kW-Labs nmecr (MIT). Fix after review: rho clamped at 0 and n' applied for all rho > 0 (deliberate deviation from nmecr's rho >= 0.5 switch), daily form rejects reporting periods over 12 months, tests added red-first; FSU constant comments and `.cursor/skills/mv-engine/references/ashrae14-formulas.md` ("Implemented choices") updated | Claude (Anthropic) | pending |
