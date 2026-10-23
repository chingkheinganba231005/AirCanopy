# Working rules for AirCanopy

These rules apply to every working session, human or AI-assisted. The plan, the progress checklist and the step-by-step log are in `JOURNAL.md`. Read it before starting work.

## The journal

1. `JOURNAL.md` is the project's memory and its teaching material. The owner is learning how the project is built, so every session:
   - adds a dated entry to Part 5 (Goal, What we did, Why, How, What to understand, Problems and fixes, Next), in simple words;
   - ticks Part 4 and updates the plan in Part 3 if it changed;
   - records decisions in Part 7 and questions for the owner in Part 8;
   - adds a lesson to Part 6 the first time a new scientific or computing idea appears.
2. Write for a capable student who has not seen the topic before: plain words first, the technical term second, a formula only when it helps.

## Commits

3. Commit and push after each meaningful, tested step. Never leave more than about an hour of work uncommitted. Push with `git push -u origin main` (retry up to four times with 2, 4, 8 and 16 second waits on network errors).
4. Commits are authored by the owner. Before the first commit of a session, check `git config user.name` (Chingkheinganba Rajkumar) and `git config user.email` (217364735+chingkheinganba231005@users.noreply.github.com) in this repository.
5. Commit messages: a short summary line, a blank line, then what changed and why.

## Numbers and claims

6. Never invent a number. Every constant, dataset and claim comes from either
   - a cited source (author, year, table or page, URL, access date), or
   - our own dated computation, written to `results/` with the code version that produced it.
7. If a source cannot be verified (for example the network blocks it), mark it `UNVERIFIED` in `docs/sources.md` and tell the owner.
8. Check the licence of every library and dataset before using it, and record it in `docs/sources.md`. Data files get their source URL, access date, licence and SHA-256 recorded next to them.
9. Every model output shown to users is labelled as simulated or predicted, with its uncertainty where we have one.

## Code

10. Python 3.11+, package in `src/aircanopy/`, dependencies managed with uv (`uv.lock` is committed). JavaScript for the viewer lives in `ui/`.
11. Plain, direct English in code, docs and messages. No filler, no marketing tone, no emoji. Comments explain why, not what.
12. Every module has tests. Physics code is checked against analytic solutions or published references before it is used for anything else.
13. Verify by running things, not by assuming: tests, benchmarks, and screenshots of the viewer that you look at.

## Environment notes

14. The cloud container used for AI-assisted sessions has no GPU: tests run on CPU with small grids. Large runs go to a GPU (Google Colab or rented), through notebooks or scripts that record what they ran.
15. In that container, the Overpass API and Open-Meteo are blocked; Amazon S3 open data, the OpenStreetMap main API, PyPI and npm are reachable. Tests never need the network: they use small stored samples.

## Commands

| Task | Command |
| --- | --- |
| Install (development) | `uv sync` |
| Lint and format check | `uv run ruff check . && uv run ruff format --check .` |
| Tests | `uv run pytest` |
| Viewer: install, format check, type check, unit tests | `cd ui && npm ci && npm run format:check && npm run typecheck && npm test` |
| Viewer: build, then browser tests (screenshots in `ui/test-results/screens/`) | `cd ui && npm run build && npm run e2e` |
| Open a site in the viewer | `uv run aircanopy view sites/NAME` |
