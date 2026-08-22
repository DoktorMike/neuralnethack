# Agent Instructions

Instructions for AI agents operating in this repository.

## Source of Truth

For all project information, refer to:

- **[README.md](README.md)** — project overview, features, quick start
- **[doc/](doc/README.md)** — deep-dive reference (this repo uses `doc/`, not `docs/`)

## Agent-Specific Instructions

- Build and test via the Makefile: `make` to build, `make test` to run all
  tests (`ctest --test-dir build --output-on-failure` for direct control).
- Never edit `single_include/neuralnethack.hh` by hand — it is generated.
  After changing library source or headers, run `make single-include` and
  commit the regenerated header; CI fails if it is out of sync.
- Run `make format` (clang-format) before committing source changes; CI
  checks formatting.
- Releases: only via `make release`. Never run standard-version directly —
  formatting and the regenerated amalgamation must land before the tag. Not
  done until pushed AND published as a GitHub release (`gh release create`)
  with hand-written notes. Details:
  [doc/development.md](doc/development.md#releases).
- Commits: Conventional Commits, caveman-commit style — imperative subject,
  no trailing period, aim ≤50 chars (hard cap 72); body only when the why
  isn't obvious from the diff; always a body for breaking changes and
  reverts; no multi-paragraph prose; no AI attribution trailers.

## Documentation Index

| Topic | Location |
|---|---|
| Architecture, design decisions, type strings, serialization | [doc/architecture.md](doc/architecture.md) |
| Full class diagrams | [doc/design/ARCHITECTURE.md](doc/design/ARCHITECTURE.md) |
| Config format, CLI tools | [doc/configuration.md](doc/configuration.md) |
| Build options, amalgamation, adding activations, releases, scripts | [doc/development.md](doc/development.md) |
| Examples | [doc/examples.md](doc/examples.md) |
| Residual connections | [doc/residual-connections.md](doc/residual-connections.md) |
| Multi-class / softmax | [doc/multiclass.md](doc/multiclass.md) |
| Adstock / MMM | [doc/adstock.md](doc/adstock.md), [doc/spec-boxed-adstock.md](doc/spec-boxed-adstock.md) |
| Uncertainty | [doc/uncertainty.md](doc/uncertainty.md) |
| Benchmarks / comparison | [doc/comparison.md](doc/comparison.md) |

## Fallback

If the documentation does not contain the information you need, use standard
conventions and ask the user.
