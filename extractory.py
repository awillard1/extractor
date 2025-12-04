#!/usr/bin/env python3
"""
extractor.py - Aho-Corasick integration + trie fallback.

Features:
- analyze, gen-wordfreq, infer-transforms, export, generate-artifacts, gen-hcmask-rules
- uses pyahocorasick (if installed) for fast substring matching
- falls back to pure-Python DictTrie if pyahocorasick is not available
- retains atomic writes, logging, and previously implemented behaviors
"""
from __future__ import annotations

import argparse
import json
import math
import logging
import os
import re
import signal
import sys
import tempfile
import unicodedata
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# ---------------------------
# Optional C-based Aho-Corasick (pyahocorasick)
# ---------------------------
try:
    import ahocorasick  # type: ignore
    AHOC_AVAILABLE = True
except Exception:
    AHOC_AVAILABLE = False
    ahocorasick = None  # type: ignore

def build_automaton(words: Iterable[str]) -> Optional[object]:
    if not AHOC_AVAILABLE:
        return None
    A = ahocorasick.Automaton()
    # words should be lowercased already (load_wordfreq does that)
    for w in words:
        if not w:
            continue
        A.add_word(w, w)
    A.make_automaton()
    return A

def automaton_find_longest(A: object, s: str, min_len: int = 3) -> Optional[Tuple[str, int, int]]:
    if A is None:
        return None
    s_low = s.lower()
    best: Optional[Tuple[str, int, int]] = None
    # iter yields (end_index, value)
    for end_idx, found in A.iter(s_low):
        length = len(found)
        if length < min_len:
            continue
        start_idx = end_idx - length + 1
        if best is None or length > len(best[0]):
            best = (found, start_idx, end_idx + 1)
    return best

# ---------------------------
# Logging / constants
# ---------------------------
LOG = logging.getLogger("extractor")
DEFAULT_LOG_LEVEL = logging.INFO

UNKNOWN_WORD_LOGP = -8.0
TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
ALNUM_RE = re.compile(r'^[A-Za-z0-9]+$')
WORD_RE = re.compile(r'[A-Za-z]{2,}')
YEAR_RE = re.compile(r'^(.*?)(19|20)\d\d$')
DIGITS_RE = re.compile(r'^(.*?)(\d+)$')
NON_ALNUM = re.compile(r'[^A-Za-z0-9]')
SEPARATOR_RE = re.compile(r'[^A-Za-z0-9]+')

LEET_MAP: Dict[str, List[str]] = {
    '4': ['a'], '@': ['a'], '0': ['o'], '1': ['l', 'i'], '3': ['e'],
    '5': ['s'], '7': ['t'], '$': ['s'], '!': ['i', 'l'], '+': ['t'], '8': ['b']
}
SIMPLE_DELEET_MAP = str.maketrans({'@': 'a', '4': 'a', '0': 'o', '1': 'l', '3': 'e',
                                  '$': 's', '+': 't', '7': 't', '!': 'i', '8': 'b'})

# ---------------------------
# Optional progress bar (tqdm)
# ---------------------------
try:
    from tqdm import tqdm as _tqdm  # type: ignore

    def maybe_progress(iterable: Optional[Iterable] = None, **kwargs):
        if iterable is None:
            return lambda it: _tqdm(it, **kwargs)
        return _tqdm(iterable, **kwargs)

    TQDM_AVAILABLE = True
except Exception:
    TQDM_AVAILABLE = False

    def maybe_progress(iterable: Optional[Iterable] = None, **kwargs):
        if iterable is None:
            return lambda it: it
        return iterable

# ---------------------------
# Signal handling
# ---------------------------
STOPPED = False

def _signal_handler(signum, frame):
    global STOPPED
    STOPPED = True
    LOG.info("received interrupt (SIGINT). Attempting graceful stop...")

signal.signal(signal.SIGINT, _signal_handler)

# ---------------------------
# Atomic write helpers
# ---------------------------
def _atomic_write_text(path: Path, text_iterable: Iterable[str], encoding: str = "utf-8") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            for line in text_iterable:
                fh.write(line)
        Path(tmp).replace(path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise
    return path

def _atomic_write_json(path: Path, obj, encoding: str = "utf-8", **json_kwargs) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, **json_kwargs)
        Path(tmp).replace(path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise
    return path

def _safe_open_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh, start=1):
            if STOPPED:
                LOG.debug("stop flag set while reading %s", path)
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                LOG.warning("skipping invalid JSON on line %d in %s: %s", i, path, e)

# ---------------------------
# Pure-Python Trie fallback
# ---------------------------
class DictTrie:
    def __init__(self):
        self.root = {}

    def build(self, words: Iterable[str]) -> None:
        root = {}
        for w in words:
            node = root
            for ch in w:
                node = node.setdefault(ch, {})
            node.setdefault("_end", True)
        self.root = root

    def find_longest_in(self, s: str, min_len: int = 3) -> Optional[Tuple[str, int, int]]:
        s = s.lower()
        best = None
        L = len(s)
        for i in range(L):
            node = self.root
            j = i
            current_match_end = -1
            while j < L and s[j] in node:
                node = node[s[j]]
                j += 1
                if node.get("_end"):
                    current_match_end = j
            if current_match_end != -1:
                length = current_match_end - i
                if length >= min_len:
                    candidate = s[i:current_match_end]
                    if best is None or len(candidate) > len(best[0]):
                        best = (candidate, i, current_match_end)
        return best

# ---------------------------
# Normalization & de-leet helpers
# ---------------------------
def normalize_unicode_lower(s: str) -> str:
    return unicodedata.normalize("NFKC", s).lower()

def simple_deleet(s: str) -> str:
    return s.translate(SIMPLE_DELEET_MAP)

def word_parts_from_password(pw: str) -> List[str]:
    return [p for p in SEPARATOR_RE.split(pw) if p]

# ---------------------------
# Wordfreq loader
# ---------------------------
def load_wordfreq(path: Optional[str]) -> Tuple[Dict[str, float], Dict[str, int]]:
    if not path:
        return {}, {}
    p = Path(path)
    if not p.exists():
        LOG.warning("dictionary file not found: %s", path)
        return {}, {}
    try:
        total = 0
        freq: Dict[str, int] = {}
        with p.open(encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                w = parts[0].lower()
                c: int = 1
                if len(parts) > 1:
                    try:
                        c = int(parts[1])
                    except Exception:
                        c = 1
                freq[w] = freq.get(w, 0) + c
                total += c
        if total == 0:
            total = 1
        logprob = {w: math.log(c / total) for w, c in freq.items()}
        return logprob, freq
    except Exception:
        LOG.exception("error loading wordfreq file %s", path)
        return {}, {}

# ---------------------------
# Candidate generation / scoring / segmentation
# ---------------------------
def candidates_from_token(token: str, beam: int = 500, max_len_for_beam: int = 64) -> List[str]:
    token = token.strip()
    if not token:
        return []
    if len(token) > max_len_for_beam and beam > 50:
        beam = max(50, beam // 4)
    heap: List[Tuple[float, str]] = [(0.0, "")]
    for ch in token:
        next_heap: List[Tuple[float, str]] = []
        subs = [ch]
        lower = ch.lower()
        if lower in LEET_MAP:
            subs += LEET_MAP[lower]
        for sc, prefix in heap:
            for s in subs:
                penalty = 0.0 if s == ch else 1.0
                next_heap.append((sc + penalty, prefix + s))
        next_heap.sort(key=lambda x: (x[0], x[1]))
        heap = next_heap[:beam]
    return [cand for _, cand in heap]

def score_candidate_by_dict(candidate: str, logprob_dict: Dict[str, float]) -> float:
    w = candidate.lower()
    if w in logprob_dict:
        return -logprob_dict[w]
    return -UNKNOWN_WORD_LOGP + len(w) * 0.5

def segment_token(token: str, logprob_dict: Dict[str, float], max_word_len: int = 30) -> Tuple[float, List[str]]:
    token_l = token.lower()
    n = len(token_l)
    if n == 0:
        return UNKNOWN_WORD_LOGP, []
    dp_score = [-1e9] * (n + 1)
    dp_seg: List[List[str]] = [[] for _ in range(n + 1)]
    dp_score[0] = 0.0
    getp = logprob_dict.get
    for i in range(n):
        if dp_score[i] < -1e8:
            continue
        for j in range(i + 1, min(n, i + max_word_len) + 1):
            piece = token_l[i:j]
            score = getp(piece, None)
            if score is None:
                if len(piece) <= 2:
                    score = UNKNOWN_WORD_LOGP
                else:
                    continue
            newscore = dp_score[i] + score
            if newscore > dp_score[j]:
                dp_score[j] = newscore
                dp_seg[j] = dp_seg[i] + [piece]
    if dp_score[n] < -1e8:
        return UNKNOWN_WORD_LOGP, [token_l]
    return dp_score[n], dp_seg[n]

# ---------------------------
# Pattern extraction & transforms
# ---------------------------
def char_class(c: str) -> str:
    if c.isalpha():
        return 'L'
    if c.isdigit():
        return 'D'
    return 'S'

def template_of(pw: str) -> str:
    if not pw:
        return ''
    cur = char_class(pw[0])
    parts = [cur]
    for ch in pw[1:]:
        cc = char_class(ch)
        if cc == cur:
            continue
        cur = cc
        parts.append(cur)
    return '-'.join(parts)

def normalize_plain(plain: str) -> str:
    return NON_ALNUM.sub('', normalize_unicode_lower(plain))

def infer_transforms_from_cracked(cracked_lines: Iterable[str], dict_set: set, automaton: Optional[object] = None, trie: Optional[DictTrie] = None) -> Dict:
    transforms = Counter()
    examples: Dict[str, str] = {}
    for plain in cracked_lines:
        if STOPPED:
            LOG.debug("stop during infer_transforms_from_cracked")
            break
        plain = plain.strip()
        if not plain:
            continue
        norm = normalize_plain(plain)
        if not norm:
            continue

        base = None
        parts = word_parts_from_password(plain)
        found = None
        for part in parts:
            pnorm = normalize_unicode_lower(part)
            if pnorm in dict_set:
                found = pnorm
                break
            pdel = simple_deleet(part)
            pdeln = normalize_unicode_lower(pdel)
            if pdeln in dict_set:
                found = pdeln
                break
            # Aho-Corasick preferred
            if automaton:
                match = automaton_find_longest(automaton, part)
                if match:
                    found = match[0]
                    break
                match2 = automaton_find_longest(automaton, pdel)
                if match2:
                    found = match2[0]
                    break
            # trie fallback
            if trie:
                match = trie.find_longest_in(part)
                if match:
                    found = match[0]
                    break
                match2 = trie.find_longest_in(pdel)
                if match2:
                    found = match2[0]
                    break

        if not found:
            if automaton:
                match = automaton_find_longest(automaton, norm)
                if match:
                    found = match[0]
            if not found and trie:
                match = trie.find_longest_in(norm)
                if match:
                    found = match[0]
            if not found:
                # brute-force fallback (previous behavior)
                L = len(norm)
                for i in range(L):
                    for j in range(L, i, -1):
                        sub = norm[i:j]
                        if sub in dict_set:
                            found = sub
                            break
                    if found:
                        break

        base = found or None
        if not base:
            m = DIGITS_RE.match(norm)
            base = m.group(1) if m else norm

        m = DIGITS_RE.match(plain)
        if m and plain.startswith(base):
            s = m.group(2)
            transforms[f"append_digits:{len(s)}"] += 1
            examples.setdefault(f"append_digits:{len(s)}", plain)
        my = YEAR_RE.match(plain)
        if my:
            transforms["append_year"] += 1
            examples.setdefault("append_year", plain)
        if plain != plain.lower() and base == base.lower():
            transforms["capitalize_first"] += 1
            examples.setdefault("capitalize_first", plain)
        if any(ch * 2 in plain for ch in set(plain)):
            transforms["repeat_char"] += 1
            examples.setdefault("repeat_char", plain)
        if len(plain) == len(base):
            diffs = sum(1 for a, b in zip(plain, base) if a.lower() != b.lower())
            if diffs > 0:
                transforms[f"subs_count:{diffs}"] += 1
                examples.setdefault(f"subs_count:{diffs}", plain)
    return {"counts": transforms.most_common(), "examples": examples}

# ---------------------------
# Export artifact helpers
# ---------------------------
def load_analysis(analysis_path: str, show_progress: bool = True) -> Tuple[List[Tuple[float, str]], List[Tuple[str, int]]]:
    candidates: List[Tuple[float, str]] = []
    templates = Counter()
    p = Path(analysis_path)
    if not p.exists():
        LOG.error("analysis file %s not found", analysis_path)
        return candidates, []
    iterator = _safe_open_jsonl(p)
    if show_progress:
        iterator = maybe_progress(iterator, desc="Reading analysis", unit="pw")
    for obj in iterator:
        if STOPPED:
            break
        tpl = obj.get("template")
        if tpl:
            templates[tpl] += 1
        for c in obj.get("candidates", []):
            candidates.append((float(c.get("rank_score", 0.0)), str(c.get("candidate", ""))))
    return candidates, sorted(templates.items(), key=lambda x: -x[1])

def write_prioritized_wordlist(candidates: List[Tuple[float, str]], out_path: str, topk: Optional[int] = None, show_progress: bool = True) -> int:
    candidates_sorted = sorted(candidates, key=lambda x: (x[0], x[1]))
    seen = set()
    written = 0
    outp = Path(out_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    def lines():
        nonlocal written
        for score, cand in candidates_sorted:
            if STOPPED:
                LOG.info("interrupted during prioritized-wordlist export; partial file saved.")
                break
            if cand in seen:
                continue
            yield cand + "\n"
            seen.add(cand)
            written += 1
            if topk and written >= topk:
                break
    _atomic_write_text(outp, lines())
    LOG.info("prioritized wordlist written to %s (%d entries)", outp, written)
    return written

def emit_masks_from_templates(template_counts: List[Tuple[str, int]], masks_dir: Path, top_n: int = 20, show_progress: bool = True) -> Tuple[Path, List[Tuple[str, str]]]:
    masks_dir.mkdir(parents=True, exist_ok=True)
    masks: List[Tuple[str, str]] = []
    iterable = template_counts[:top_n]
    if show_progress:
        iterable = maybe_progress(iterable, desc="Generating masks", unit="masks")
    for tpl, _ in iterable:
        if STOPPED:
            LOG.info("interrupted during mask generation; partial results saved.")
            break
        parts = tpl.split('-') if tpl else []
        mask = "".join("?l" if p == "L" else "?d" if p == "D" else "?s" for p in parts)
        if not mask:
            continue
        safe_name = tpl.replace('-', '_') or 'empty'
        fname = masks_dir / f"{safe_name}.mask"
        _atomic_write_text(fname, (mask + "\n",))
        masks.append((str(fname), mask))
    hcmask_path = masks_dir.parent / "masks.hcmask"
    _atomic_write_text(hcmask_path, (m + "\n" for _, m in masks))
    LOG.info("masks written to %s (%d masks)", hcmask_path, len(masks))
    return hcmask_path, masks

def emit_rules_from_transforms(transforms: Dict, rules_dir: Path, show_progress: bool = True) -> List[str]:
    rules_dir.mkdir(parents=True, exist_ok=True)
    rules_out: List[str] = []
    counts = transforms.get("counts", []) if transforms else []
    iterable = counts
    if show_progress:
        iterable = maybe_progress(counts, desc="Generating transform-based rules", unit="rules")
    for name, _ in iterable:
        if STOPPED:
            LOG.info("interrupted during rule generation; partial results saved.")
            break
        if name == "capitalize_first":
            fname = rules_dir / "capitalize_first.rule"
            _atomic_write_text(fname, ("u\n",))
            rules_out.append(str(fname))
        elif name.startswith("append_digits:"):
            try:
                n = int(name.split(":", 1)[1])
            except Exception:
                n = 1
            fname = rules_dir / f"append_digits_len{n}.rule"
            _atomic_write_text(fname, (f"# append {n} digits: prefer combinator (-a 1) with suffixes/digits_len{n}.txt\n",))
            rules_out.append(str(fname))
        elif name == "append_year":
            fname = rules_dir / "append_year.rule"
            _atomic_write_text(fname, ("# use combinator with suffixes/years.txt\n",))
            rules_out.append(str(fname))
        else:
            safe = re.sub(r'[^\w\-_\.]', '_', name)
            fname = rules_dir / f"{safe}.rule"
            _atomic_write_text(fname, (f"# placeholder for {name}\n",))
            rules_out.append(str(fname))
    LOG.info("generated %d rule files in %s", len(rules_out), rules_dir)
    return rules_out

def emit_suffixes_from_transforms(transforms: Dict, suffixes_dir: Path, max_digit_len: int = 4, show_progress: bool = True) -> List[str]:
    suffixes_dir.mkdir(parents=True, exist_ok=True)
    created: List[str] = []
    counts = transforms.get("counts", []) if transforms else []
    iterable = counts
    if show_progress:
        iterable = maybe_progress(counts, desc="Generating suffix lists", unit="suffix-types")
    for name, _ in iterable:
        if STOPPED:
            LOG.info("interrupted during suffix generation; partial results saved.")
            break
        if name.startswith("append_digits:"):
            try:
                n = int(name.split(":", 1)[1])
            except Exception:
                n = 1
            ncap = min(n, max_digit_len)
            fname = suffixes_dir / f"digits_len{ncap}.txt"
            def gen():
                if ncap <= 3:
                    for i in range(10 ** ncap):
                        yield f"{i:0{ncap}d}\n"
                else:
                    for i in range(1000):
                        yield f"{i:03d}\n"
            _atomic_write_text(fname, gen())
            created.append(str(fname))
        if name == "append_year":
            fname = suffixes_dir / "years.txt"
            def gen_years():
                for y in range(1950, 2031):
                    yield str(y) + "\n"
            _atomic_write_text(fname, gen_years())
            created.append(str(fname))
    LOG.info("created %d suffix files in %s", len(created), suffixes_dir)
    return created

def combine_rule_files(rules_dir: Path, out_file: Optional[Path] = None, show_progress: bool = True) -> Tuple[Optional[Path], List[str], int]:
    rules_dir = Path(rules_dir)
    if out_file is None:
        out_file = rules_dir / "all_rules.rule"
    rule_files = sorted([p for p in rules_dir.glob("*.rule") if p.is_file()], key=lambda p: p.name)
    if not rule_files:
        LOG.warning("no .rule files found in %s", rules_dir)
        return None, [], 0

    preferred = ["prepend_affixes.rule", "capitalize_first.rule", "append_affixes.rule"]
    files_map = {f.name: f for f in rule_files}
    ordered: List[Path] = []
    for name in preferred:
        if name in files_map:
            ordered.append(files_map.pop(name))
    for f in sorted(files_map.values(), key=lambda x: x.name):
        ordered.append(f)

    seen = OrderedDict()
    sources: List[str] = []
    iterable = ordered
    if show_progress:
        iterable = maybe_progress(ordered, desc="Scanning rule files", unit="files")
    for rf in iterable:
        sources.append(rf.name)
        try:
            with rf.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.rstrip("\n\r")
                    if not line:
                        continue
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    normalized = " ".join(stripped.split())
                    if normalized not in seen:
                        seen[normalized] = rf.name
        except Exception as e:
            LOG.warning("can't read rule file %s: %s", rf, e)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    def out_lines():
        yield f"# Combined rules file generated from: {', '.join(sources)}\n"
        yield f"# Total unique rules: {len(seen)}\n\n"
        for rule_line in seen.keys():
            if STOPPED:
                LOG.info("interrupted during combined-rule write; partial file saved.")
                break
            yield rule_line + "\n"

    _atomic_write_text(out_file, out_lines())
    LOG.info("combined rules written to %s (%d unique rules)", out_file, len(seen))
    return out_file, sources, len(seen)

# ---------------------------
# gen-hcmask-rules (combined)
# ---------------------------
def gen_hcmask_and_rules_from_analysis(analysis_path: str, transforms_path: Optional[str], templates_path: Optional[str], out_dir: str, top_affixes: int = 200, top_masks: int = 50, show_progress: bool = True) -> Dict:
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    entries: List[dict] = []
    p = Path(analysis_path)
    if not p.exists():
        LOG.error("analysis file %s not found", analysis_path)
        raise FileNotFoundError(analysis_path)

    iterator = _safe_open_jsonl(p)
    if show_progress:
        iterator = maybe_progress(iterator, desc="Reading analysis for hcmask/rules", unit="pw")
    for obj in iterator:
        if STOPPED:
            break
        entries.append(obj)

    tpl_counts: List[Tuple[str, int]] = []
    if templates_path and Path(templates_path).exists():
        try:
            with open(templates_path, encoding='utf-8') as fh:
                tpl_list = json.load(fh)
                tpl_counts = [(t[0], int(t[1])) for t in tpl_list]
        except Exception:
            LOG.warning("can't load templates file %s, will infer from analysis", templates_path)
            tpl_counts = []
    if not tpl_counts:
        tpl_counter = Counter()
        for e in entries:
            tpl = e.get("template")
            if tpl:
                tpl_counter[tpl] += 1
        tpl_counts = tpl_counter.most_common()

    hcmask_path, masks_written = emit_masks_from_templates(tpl_counts, outdir / "masks", top_n=top_masks, show_progress=show_progress)

    prefixes = Counter()
    suffixes = Counter()
    iterable = entries
    if show_progress:
        iterable = maybe_progress(entries, desc="Scanning entries for affixes", unit="pw")
    for obj in iterable:
        if STOPPED:
            break
        orig = obj.get("orig", "")
        if not orig:
            continue
        m = WORD_RE.search(orig)
        if not m:
            de = simple_deleet(orig)
            m = WORD_RE.search(de)
            if not m:
                continue
            else:
                s, e = m.span()
        else:
            s, e = m.span()
        prefix = orig[:s]
        suffix = orig[e:]
        if prefix and len(prefix) <= 8 and ALNUM_RE.match(prefix):
            prefixes[prefix] += 1
        if suffix and len(suffix) <= 8 and ALNUM_RE.match(suffix):
            suffixes[suffix] += 1

    prefixes_top = prefixes.most_common(top_affixes)
    suffixes_top = suffixes.most_common(top_affixes)

    rules_dir = outdir / "rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    append_file = rules_dir / "append_affixes.rule"
    def gen_append():
        for affix, cnt in suffixes_top:
            if STOPPED:
                break
            if not ALNUM_RE.match(affix):
                continue
            rule_line = "".join("$" + ch for ch in affix)
            yield rule_line + "\n"
    _atomic_write_text(append_file, gen_append())

    prepend_file = rules_dir / "prepend_affixes.rule"
    def gen_prepend():
        for affix, cnt in prefixes_top:
            if STOPPED:
                break
            if not ALNUM_RE.match(affix):
                continue
            rev = affix[::-1]
            rule_line = "".join("^" + ch for ch in rev)
            yield rule_line + "\n"
    _atomic_write_text(prepend_file, gen_prepend())

    cap_written = None
    if transforms_path and Path(transforms_path).exists():
        try:
            with open(transforms_path, encoding='utf-8') as fh:
                transforms = json.load(fh)
            counts = dict(transforms.get("counts", []))
            if "capitalize_first" in counts:
                cap_file = rules_dir / "capitalize_first.rule"
                _atomic_write_text(cap_file, ("u\n",))
                cap_written = str(cap_file)
        except Exception:
            LOG.warning("can't read transforms file %s; skipping capitalize rule", transforms_path)

    combined_path, sources, unique_count = combine_rule_files(rules_dir, out_file=rules_dir / "all_rules.rule", show_progress=show_progress)

    result = {
        "hcmask": str(hcmask_path),
        "masks_count": len(masks_written),
        "append_rule": str(append_file),
        "prepend_rule": str(prepend_file),
        "capitalize_rule": cap_written,
        "combined_rules": str(combined_path) if combined_path else None,
        "combined_sources": sources,
        "combined_unique_count": unique_count
    }
    LOG.info("gen_hcmask_and_rules_from_analysis completed.")
    return result

# ---------------------------
# export_artifacts
# ---------------------------
def export_artifacts(analysis_jsonl: str, transforms_json: Optional[str], templates_json: Optional[str], out_dir: str, topk: Optional[int] = None, show_progress: bool = True) -> Dict:
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    candidates, template_counts = load_analysis(analysis_jsonl, show_progress=show_progress)

    wordlist_path = outdir / "prioritized_wordlist.txt"
    n = write_prioritized_wordlist(candidates, str(wordlist_path), topk=topk, show_progress=show_progress)

    tpl_counts = []
    if templates_json and Path(templates_json).exists():
        try:
            with open(templates_json, encoding='utf-8') as fh:
                tpl_list = json.load(fh)
                tpl_counts = [(t[0], int(t[1])) for t in tpl_list]
        except Exception:
            LOG.warning("can't load templates.json %s; falling back to internal", templates_json)
            tpl_counts = template_counts
    else:
        tpl_counts = template_counts

    hcmask_path, masks = emit_masks_from_templates(tpl_counts, outdir / "masks", top_n=50, show_progress=show_progress)

    transforms = None
    if transforms_json and Path(transforms_json).exists():
        try:
            with open(transforms_json, encoding='utf-8') as fh:
                transforms = json.load(fh)
        except Exception:
            LOG.warning("can't parse transforms json %s", transforms_json)
            transforms = None

    rules = emit_rules_from_transforms(transforms or {}, outdir / "rules", show_progress=show_progress)
    suffixes = emit_suffixes_from_transforms(transforms or {}, outdir / "suffixes", show_progress=show_progress)

    combined = None
    rules_dir = outdir / "rules"
    if rules_dir.exists():
        combined_path, sources, unique_count = combine_rule_files(rules_dir, out_file=rules_dir / "all_rules.rule", show_progress=show_progress)
        combined = str(combined_path) if combined_path else None

    res = {
        "wordlist": str(wordlist_path),
        "wordlist_count": n,
        "masks_hcmask": str(hcmask_path),
        "masks_count": len(masks),
        "rules": rules,
        "suffixes": suffixes,
        "combined_rules": combined
    }
    LOG.info("export_artifacts completed.")
    return res

# ---------------------------
# CLI wiring and analyze/infer helpers
# ---------------------------
def _add_export_subparsers(sp: argparse._SubParsersAction):
    ap_export = sp.add_parser("export", help="Export Hashcat artifacts from analysis + transforms")
    ap_export.add_argument("--analysis", required=True, help="analysis.jsonl from analyze")
    ap_export.add_argument("--transforms", required=False, help="transforms.json from infer-transforms")
    ap_export.add_argument("--templates", required=False, help="templates.json from analyze (optional)")
    ap_export.add_argument("--out-dir", required=True, help="output directory for artifacts")
    ap_export.add_argument("--topk", type=int, default=None, help="max candidates for prioritized_wordlist.txt")

    ap_genart = sp.add_parser("generate-artifacts", help="Run export AND then gen-hcmask-rules (combined workflow)")
    ap_genart.add_argument("--analysis", required=True, help="analysis.jsonl from analyze")
    ap_genart.add_argument("--transforms", required=False, help="transforms.json from infer-transforms")
    ap_genart.add_argument("--templates", required=False, help="templates.json from analyze (optional)")
    ap_genart.add_argument("--out-dir", required=True, help="output directory for artifacts")
    ap_genart.add_argument("--topk", type=int, default=None, help="max candidates for prioritized_wordlist.txt")

def analyze_passwords(pw_list_path: str, dict_path: Optional[str], out_dir: str, topk_per_pw: int = 5, auto_gen_dict: bool = False, gen_params: Optional[dict] = None, show_progress: bool = True) -> Tuple[Path, Path, List[Tuple[float, str]], List[Tuple[str, int]]]:
    if gen_params is None:
        gen_params = {}

    pw_path = Path(pw_list_path)
    if not pw_path.exists():
        LOG.error("password list %s does not exist", pw_list_path)
        raise FileNotFoundError(pw_list_path)

    if not dict_path and auto_gen_dict:
        auto_path = Path(out_dir) / "wordfreq.auto.txt"
        LOG.info("auto-generating wordfreq from %s -> %s", pw_list_path, auto_path)
        gen_wordfreq_from_file(str(pw_path), out_path=str(auto_path), min_token_len=gen_params.get("min_token_len", 2), min_count=gen_params.get("min_count", 1), top_n=gen_params.get("top_n", None), sample_lines=gen_params.get("sample_lines", None), show_progress=show_progress)
        dict_path = str(auto_path)

    logprob_dict, freq = load_wordfreq(dict_path)
    template_counter = Counter()
    analysis_out = Path(out_dir) / "analysis.jsonl"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    candidates_master: List[Tuple[float, str]] = []

    # build dict lookup: prefer Aho automaton if available
    automaton = None
    trie = None
    if freq:
        if AHOC_AVAILABLE:
            automaton = build_automaton(freq.keys())
            LOG.info("pyahocorasick automaton built with %d words", len(freq))
        else:
            trie = DictTrie()
            trie.build(freq.keys())
            LOG.info("fallback DictTrie built with %d words", len(freq))

    with pw_path.open(encoding='utf-8', errors='ignore') as fh_in, analysis_out.open("w", encoding='utf-8') as fh_out:
        iterator = fh_in
        if show_progress:
            iterator = maybe_progress(fh_in, desc="Analyzing passwords", unit="pw")
        for line in iterator:
            if STOPPED:
                LOG.info("interrupted during analysis; writing partial analysis file.")
                break
            pw = line.rstrip("\n")
            if not pw:
                continue
            tpl = template_of(pw)
            template_counter[tpl] += 1

            parts = word_parts_from_password(pw)
            raw_cands = set()
            for c in candidates_from_token(pw, beam=gen_params.get("beam", 500)):
                raw_cands.add(c)
            for part in parts:
                if not part:
                    continue
                for c in candidates_from_token(part, beam=max(50, gen_params.get("beam", 500)//5)):
                    raw_cands.add(c)

            scored: List[Tuple[float, str, float, List[str]]] = []
            for c in raw_cands:
                if STOPPED:
                    break
                seg_score, seg = segment_token(c, logprob_dict)
                dict_score = score_candidate_by_dict(c, logprob_dict)
                combined = dict_score - seg_score
                scored.append((combined, c, seg_score, seg))
            scored.sort(key=lambda x: (x[0], x[1]))
            top = scored[:topk_per_pw]
            entry = {
                "orig": pw,
                "template": tpl,
                "candidates": [
                    {"candidate": c, "rank_score": float(sc), "seg_score": float(ss), "segmentation": seg}
                    for sc, c, ss, seg in top
                ]
            }
            for sc, c, ss, seg in top:
                candidates_master.append((sc, c))
            fh_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    templates_out = Path(out_dir) / "templates.json"
    total = sum(template_counter.values()) or 1
    tpl_list = [(k, v, v / total) for k, v in template_counter.most_common()]
    _atomic_write_json(templates_out, tpl_list)

    LOG.info("analysis written to %s", analysis_out)
    return analysis_out, templates_out, candidates_master, template_counter.most_common()

def infer_transforms_cmd(cracked_path: str, dict_path: str, out_dir: str, show_progress: bool = True) -> Tuple[str, dict]:
    cracked = Path(cracked_path)
    if not cracked.exists():
        raise FileNotFoundError(cracked_path)
    dictp = Path(dict_path)
    if not dictp.exists():
        raise FileNotFoundError(dict_path)
    outdir = Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    _, freq = load_wordfreq(str(dictp))
    dict_set = set(freq.keys())

    automaton = None
    trie = None
    if freq:
        if AHOC_AVAILABLE:
            automaton = build_automaton(freq.keys())
            LOG.info("pyahocorasick automaton built with %d words", len(freq))
        else:
            trie = DictTrie()
            trie.build(freq.keys())
            LOG.info("fallback DictTrie built with %d words", len(freq))

    lines = []
    with cracked.open(encoding='utf-8', errors='ignore') as fh:
        iterator = fh
        if show_progress:
            iterator = maybe_progress(fh, desc="Reading cracked list", unit="lines")
        for line in iterator:
            if STOPPED:
                break
            lines.append(line.strip())

    transforms = infer_transforms_from_cracked(lines, dict_set, automaton=automaton, trie=trie)
    out_file = outdir / "transforms.json"
    _atomic_write_json(out_file, transforms)
    LOG.info("transforms written to %s", out_file)
    return str(out_file), transforms

# ---------------------------
# CLI main
# ---------------------------
def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=DEFAULT_LOG_LEVEL, format="%(levelname)s: %(message)s")
    p = argparse.ArgumentParser(prog="extractor.py")
    sp = p.add_subparsers(dest="cmd")
    try:
        sp.required = True
    except Exception:
        pass

    ap_analyze = sp.add_parser("analyze", help="Analyze a password list")
    ap_analyze.add_argument("--pw-list", required=True)
    ap_analyze.add_argument("--dict", required=False, default=None)
    ap_analyze.add_argument("--out-dir", required=True)
    ap_analyze.add_argument("--topk-per-pw", type=int, default=5)
    ap_analyze.add_argument("--auto-gen-dict", action="store_true")
    ap_analyze.add_argument("--gen-min-token-len", type=int, default=2)
    ap_analyze.add_argument("--gen-min-count", type=int, default=1)
    ap_analyze.add_argument("--gen-top-n", type=int, default=None)
    ap_analyze.add_argument("--gen-sample-lines", type=int, default=None)
    ap_analyze.add_argument("--beam", type=int, default=500, help="beam width for candidate generation")
    ap_analyze.add_argument("--no-progress", action="store_true")

    ap_gen = sp.add_parser("gen-wordfreq", help="Generate wordfreq from text")
    ap_gen.add_argument("--input", required=True)
    ap_gen.add_argument("--out", required=False, default="wordfreq.txt")
    ap_gen.add_argument("--min-token-len", type=int, default=2)
    ap_gen.add_argument("--min-count", type=int, default=1)
    ap_gen.add_argument("--top", type=int, default=None)
    ap_gen.add_argument("--sample-lines", type=int, default=None)
    ap_gen.add_argument("--no-progress", action="store_true")

    ap_infer = sp.add_parser("infer-transforms", help="Infer transforms from cracked list")
    ap_infer.add_argument("--cracked", required=True)
    ap_infer.add_argument("--dict", required=True)
    ap_infer.add_argument("--out-dir", required=True)
    ap_infer.add_argument("--no-progress", action="store_true")

    _add_export_subparsers(sp)

    ap_hcmask = sp.add_parser("gen-hcmask-rules", help="Generate masks.hcmask and Hashcat .rule files")
    ap_hcmask.add_argument("--analysis", required=True, help="analysis.jsonl from analyze")
    ap_hcmask.add_argument("--transforms", required=False, help="transforms.json from infer-transforms (optional)")
    ap_hcmask.add_argument("--templates", required=False, help="templates.json from analyze (optional)")
    ap_hcmask.add_argument("--out-dir", required=True, help="output directory for artifacts")
    ap_hcmask.add_argument("--top-affixes", type=int, default=200, help="how many affixes to consider")
    ap_hcmask.add_argument("--top-masks", type=int, default=50, help="how many masks to include")
    ap_hcmask.add_argument("--no-progress", action="store_true")

    args = p.parse_args(argv)

    def _get_outdir(namespace):
        return getattr(namespace, "out_dir", None) or getattr(namespace, "outdir", None) or getattr(namespace, "out-dir", None)

    try:
        if args.cmd == "analyze":
            show_progress = not getattr(args, "no_progress", False)
            gen_params = {"min_token_len": args.gen_min_token_len, "min_count": args.gen_min_count, "top_n": args.gen_top_n, "sample_lines": args.gen_sample_lines, "beam": args.beam}
            analysis_out, templates_out, candidates_master, template_counts = analyze_passwords(args.pw_list, args.dict, args.out_dir, topk_per_pw=args.topk_per_pw, auto_gen_dict=args.auto_gen_dict, gen_params=gen_params, show_progress=show_progress)
            LOG.info("analysis written to %s", analysis_out)
            LOG.info("templates summary written to %s", templates_out)
            LOG.info("collected %d candidate rows", len(candidates_master))

        elif args.cmd == "gen-wordfreq":
            show_progress = not getattr(args, "no_progress", False)
            outp = gen_wordfreq_from_file(args.input, out_path=args.out, min_token_len=args.min_token_len, min_count=args.min_count, top_n=args.top, sample_lines=args.sample_lines, show_progress=show_progress)
            LOG.info("wordfreq written to: %s", outp)

        elif args.cmd == "infer-transforms":
            show_progress = not getattr(args, "no_progress", False)
            trans_path, trans_obj = infer_transforms_cmd(args.cracked, args.dict, args.out_dir, show_progress=show_progress)
            LOG.info("transforms written to %s", trans_path)

        elif args.cmd == "export":
            show_progress = True
            analysis = getattr(args, "analysis", None)
            transforms = getattr(args, "transforms", None)
            templates = getattr(args, "templates", None)
            outdir = _get_outdir(args)
            if not outdir:
                LOG.error("missing --out-dir for export command")
                sys.exit(2)
            res = export_artifacts(analysis, transforms, templates, outdir, topk=getattr(args, "topk", None), show_progress=show_progress)
            LOG.info("artifacts generated:")
            print(json.dumps(res, indent=2))

        elif args.cmd == "generate-artifacts":
            show_progress = True
            analysis = getattr(args, "analysis", None)
            transforms = getattr(args, "transforms", None)
            templates = getattr(args, "templates", None)
            outdir = _get_outdir(args)
            if not outdir:
                LOG.error("missing --out-dir for generate-artifacts command")
                sys.exit(2)
            res_export = export_artifacts(analysis, transforms, templates, outdir, topk=getattr(args, "topk", None), show_progress=show_progress)
            res_hcmask = gen_hcmask_and_rules_from_analysis(analysis, transforms, templates, outdir, top_affixes=200, top_masks=50, show_progress=show_progress)
            merged = res_export.copy()
            merged.update(res_hcmask)
            LOG.info("export + hcmask/rules generated:")
            print(json.dumps(merged, indent=2))

        elif args.cmd == "gen-hcmask-rules":
            show_progress = not getattr(args, "no_progress", False)
            outdir = _get_outdir(args)
            if not outdir:
                LOG.error("missing --out-dir for gen-hcmask-rules command")
                sys.exit(2)
            res = gen_hcmask_and_rules_from_analysis(args.analysis, args.transforms, args.templates, outdir, top_affixes=args.top_affixes, top_masks=args.top_masks, show_progress=show_progress)
            LOG.info("hcmask & rules generated:")
            print(json.dumps(res, indent=2))

        else:
            p.print_help()

    except KeyboardInterrupt:
        LOG.info("interrupted by user (Ctrl-C). Exiting.")
        sys.exit(130)
    except Exception:
        LOG.exception("fatal error")
        raise

# gen-wordfreq implementation (kept near bottom)
def gen_wordfreq_from_file(input_path: str, out_path: Optional[str] = None, min_token_len: int = 1, min_count: int = 1, top_n: Optional[int] = None, sample_lines: Optional[int] = None, show_progress: bool = True) -> Optional[str]:
    inp = Path(input_path)
    if not inp.exists():
        LOG.error("input file %s does not exist", input_path)
        raise FileNotFoundError(input_path)

    counts = Counter()
    read = 0
    try:
        with inp.open(encoding='utf-8', errors='ignore') as fh:
            iterator = fh
            if show_progress:
                try:
                    iterator = maybe_progress(fh, desc="Scanning input")
                except Exception:
                    iterator = fh
            for line in iterator:
                if STOPPED:
                    LOG.info("interrupted during wordfreq generation; writing partial results.")
                    break
                if sample_lines and read >= sample_lines:
                    break
                read += 1
                for tok in TOKEN_RE.findall(line):
                    t = tok.lower()
                    if len(t) < min_token_len:
                        continue
                    counts[t] += 1
    except Exception:
        LOG.exception("failed while reading input file")
        raise

    if min_count > 1:
        counts = Counter({w: c for w, c in counts.items() if c >= min_count})

    most = counts.most_common(top_n)
    if out_path:
        outp = Path(out_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        def lines():
            for w, c in most:
                yield f"{w} {c}\n"
        _atomic_write_text(outp, lines())
        LOG.info("wordfreq written to %s (%d entries)", outp, len(most))
        return str(outp)
    else:
        return most

if __name__ == "__main__":
    main()
