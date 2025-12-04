# extractor.py

**Password-list analysis & Hashcat artifact generator**

This repository contains `extractor.py` — a robust Python utility to analyze password lists, infer common password transforms, and export Hashcat-friendly artifacts such as prioritized candidate lists, masks (`.hcmask`), suffix lists, and `.rule` files.

The script supports two dictionary substring engines:

- **pyahocorasick** (recommended): a fast C-based Aho–Corasick automaton (optional dependency).
- **DictTrie** (built-in): a pure-Python trie fallback when `pyahocorasick` is not available.

---

## Features

- Analyze password lists and produce `analysis.jsonl` containing candidate de-leeted variants and segmentations.
- Generate a `wordfreq` from raw text inputs.
- Infer common transforms (suffixes, years, capitalization, substitutions) from a cracked password list.
- Export Hashcat artifacts: prioritized wordlist, masks, suffix lists, and `.rule` files.
- Combined workflow `generate-artifacts` to run export + generate masks / rules.
- Atomic file writes, progress indicators (when `tqdm` installed), and graceful SIGINT handling.

---

## Installation

Recommended: use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate   # or .\.venv\Scripts\activate on Windows
python -m pip install --upgrade pip
```

Optional (recommended for speed):

```bash
python -m pip install pyahocorasick tqdm
```

`pyahocorasick` provides the fastest dictionary substring matching; the script works without it using the built-in trie.

---

## Usage (CLI)

Save the script as `extractor.py` and run:

```bash
python extractor.py <command> [options]
```

Available commands:

- `analyze` — Analyze a password list and write `analysis.jsonl` + `templates.json`.
- `gen-wordfreq` — Generate a `wordfreq.txt` file from sample text.
- `infer-transforms` — Infer transforms from a cracked list using a dictionary.
- `export` — Export masks, rules, prioritized wordlist, and suffix files from `analysis.jsonl` and `transforms.json`.
- `generate-artifacts` — Run `export` and then generate masks/.rule files (combined workflow).
- `gen-hcmask-rules` — Generate `masks.hcmask` and Hashcat `.rule` files from `analysis.jsonl`.

### Common example

Analyze a password list:

```bash
python extractor.py analyze --pw-list path/to/passwords.txt --dict wordfreq.txt --out-dir out/analysis --beam 500 --topk-per-pw 5
```

Generate artifacts (export + rules):

```bash
python extractor.py generate-artifacts --analysis out/analysis/analysis.jsonl --transforms out/transforms.json --templates out/analysis/templates.json --out-dir out/artifacts
```

Generate a wordfreq from a corpus:

```bash
python extractor.py gen-wordfreq --input samples/corpus.txt --out wordfreq.txt --min-token-len 2
```

Infer transforms from a cracked file:

```bash
python extractor.py infer-transforms --cracked cracked.txt --dict wordfreq.txt --out-dir out/transforms
```

---

## Important CLI options

- `--pw-list` : path to password list for `analyze`.
- `--dict` : path to `wordfreq` dictionary used for scoring/segmentation.
- `--out-dir` : output directory for generated artifacts.
- `--topk-per-pw` : how many top candidates to keep per password (default: 5).
- `--beam` : candidate-generation beam width (default: 500). Lowering this speeds execution with modest coverage loss.
- `--no-progress` : disable progress bars.

---

## Output files

When running `analyze` and `export`, the following outputs are produced (under the `--out-dir`):

- `analysis.jsonl` — one JSON object per password with `orig`, `template`, and `candidates` (candidate, rank_score, segmentation).
- `templates.json` — aggregated template counts and frequencies.
- `prioritized_wordlist.txt` — deduplicated prioritized candidate list (used for targeted cracking).
- `masks/` and `masks.hcmask` — Hashcat masks derived from templates.
- `rules/` — generated `.rule` files (capitalize, append/prepend affixes, etc.).
- `suffixes/` — generated suffix lists (e.g., digits, years).
- `rules/all_rules.rule` — combined deduplicated rule file.

---

## Performance tips

- Install `pyahocorasick` for much faster substring matching:

```bash
python -m pip install pyahocorasick
```

- Run `analyze` with a reduced `--beam` and `--topk-per-pw` for large lists, e.g. `--beam 200 --topk-per-pw 3`.
- Use `--no-progress` when running in non-interactive environments.
- Consider parallelizing large analyses by splitting the input file and running multiple `analyze` jobs concurrently (the script is streaming-safe).

---

## Development & testing

- The script is implemented in pure Python (3.8+) and avoids non-standard dependencies except optional ones listed above.
- Logging is available; run with `python extractor.py ...` and inspect log output.

Suggested test workflow:

1. Build a small `wordfreq.txt` (or use `gen-wordfreq` on a small corpus).
2. Create a tiny `passwords.txt` containing known test cases (e.g., `john.doe1990`, `p@ssword123`, `adm1n`).
3. Run `analyze` and inspect `analysis.jsonl` to verify segmentation and candidate generation.

---

## Troubleshooting

- If the script reports `analysis.jsonl` not found, ensure `--out-dir` is writable and the `analyze` step completed without interruption.
- If substring/word detection seems weak, check that the `--dict` (`wordfreq.txt`) contains relevant tokens (names, brands, etc.).
- If you see performance issues, try installing `pyahocorasick` and decreasing `--beam`.

---

## Contributing

Contributions are welcome. Suggested improvements:

- Add an optional multiprocessing/parallel mode for `analyze` (safe chunked processing + deterministic merge).
- Add optional integration with `rapidfuzz` for fuzzy substring matching.
- Provide a small test harness (unit tests) and CI configuration.

When contributing, provide tests and maintain backward-compatible CLI behavior.

---

## License

This project is provided "as-is". No license is included by default — please add a `LICENSE` file if you intend to redistribute under a specific license.

---

## Changelog

- v1.0 — Baseline: analyze, gen-wordfreq, infer-transforms, export, gen-hcmask-rules.
- v1.x — Added pyahocorasick support (recommended) and DictTrie fallback, atomic writes, and de-leet/tokenization improvements.

---

## Contact

For questions or requests, open an issue or contact the maintainer.

