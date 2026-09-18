# Repository Guidelines

## Project Structure & Module Organization

`main.py` is the offline image-sequence entry point; `main_realtime.py` runs the RealSense workflow. Core SLAM code is in `vggt_slam/`: keep map, graph, camera, solver, submap, loop-closure, and visualization responsibilities in their corresponding modules. Evaluation runners and log-processing helpers live in `evals/`. Images and README media belong in `assets/`; `office_loop.zip` is the bundled demo data. Third-party dependencies downloaded by `setup.sh` are intentionally ignored—do not commit them.

## Setup, Run, and Evaluation Commands

Use Python 3.11 in the vggt-slam Conda environment. To activate it, run

```bash
conda activate vggt-slam
```

Then install project and external dependencies:

```bash
chmod +x setup.sh && ./setup.sh
python3 main.py --image_folder office_loop --max_loops 1 --vis_map
python3 main_realtime.py --vis_map
./evals/eval_tum.sh 32
python3 evals/process_logs_tum.py --submap_size 32
```

The setup script fetches required third-party repositories. Unzip `office_loop.zip` before the demo command. The evaluation script expects its dataset path configuration to be set first; use its output processor after a run. Use `python3 main.py --help` to inspect runtime options.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, `snake_case` for functions, variables, modules, and CLI flags, and `PascalCase` for classes (for example, `Solver` and `Submap`). Keep imports grouped as standard library, third-party, then local modules. Prefer focused helpers in the relevant `vggt_slam` module rather than expanding an entry script. Preserve the established argument names and add concise `argparse` help text for new options. No formatter or linter is configured; avoid unrelated reformatting.

## Testing Guidelines

There is currently no checked-in unit-test suite or coverage target. For changes, run the narrowest practical smoke test: use `--help` for CLI edits, the bundled office-loop demo for offline-pipeline changes, and the relevant evaluation command for metric/log changes. If adding tests, place them under `tests/`, name files `test_*.py`, and keep fixtures small and hardware-independent.

## Commit & Pull Request Guidelines

Recent history uses short, imperative summaries such as `adding real time code` and `Fix regex to handle decimal numbers in filenames`. Write a concise imperative subject that names the affected behavior; avoid generic messages such as `temp`. Pull requests should explain the motivation and validation performed, link relevant issues or papers when applicable, and include screenshots or viewer captures for visualization changes. Do not commit datasets, model weights, generated logs, point clouds, or downloaded dependencies.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For any task that requires understanding, inspecting, or modifying the codebase, first run `graphify query "<task or question>"` when `graphify-out/graph.json` exists, before browsing source files. This includes implementation tasks, debugging, refactoring, code review, and codebase questions. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts.
- Dirty `graphify-out/` files are expected after hooks or incremental updates; dirty graph files are not a reason to skip Graphify. Only skip Graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If `graphify-out/wiki/index.md` exists, use it for broad navigation instead of raw source browsing.
- Read `graphify-out/GRAPH_REPORT.md` only for broad architecture review or when query/path/explain do not surface enough context.
- After any source-code modifications, run `graphify update .` before completing the task (AST-only, no API cost). This also applies when code modifications were performed by a subagent.
