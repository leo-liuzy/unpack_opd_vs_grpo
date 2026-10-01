#!/usr/bin/env python3
"""Recover validation/metrics.json for an eval run from verl's stdout.

A val_only pass computes the full val-core/val-aux metric dict but only prints it:
trainer.validation_data_dir persists the generations (0.jsonl) and nothing else. This
script writes <out_dir>/validation/metrics.json next to them; scripts/eval_seeds.sh
reads it for the summary table and to skip evals that already finished.

Numbers are lifted verbatim from what verl printed rather than recomputed from 0.jsonl,
so the bootstrap estimators behind best@k / maj@k / worst@k stay identical to the
in-training metrics instead of depending on a reimplementation here.

Parsing note: verl emits the metrics twice. The console logger writes one flat
"step:0 - key:value - key:value ..." line, which is complete and unambiguous, so that is
the primary source. The other copy is a pprint of the dict, wrapped into adjacent quoted
fragments that routinely split a key from its value ("...'val-aux/AIME24/acc/std@16': " /
"0.083..., ..."); it is only used as a fallback, with fragment interiors concatenated
back together first.

Usage:
    # every eval in a sweep log, attributed by the sweep's ">>> evaluating ... -> " marker
    python3 write_eval_metrics.py --sweep-log /path/to/eval_sweep.log [--dry-run] [--force]

    # a single eval whose stdout was captured on its own
    python3 write_eval_metrics.py --eval-log <dir>/eval.log --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

# ">>> [18:39:12Z] evaluating <run> @ step 20 -> /path/to/eval_step20_trainprompt_think_n16"
MARKER_RE = re.compile(r">>> \[[^\]]*\] evaluating (?P<tag>.+?) -> (?P<out_dir>\S+)\s*$")
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
RAY_PREFIX_RE = re.compile(r"\((?:[A-Za-z_][\w.]*\s+)?pid=\d+[^)]*\)\s*")
METRIC_RE = re.compile(
    r"""['"](?P<key>val-(?:core|aux)/[^'"]+)['"]:\s*"""
    r"""(?P<val>-?(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|nan|inf))"""
)
NUMBER = r"-?(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|nan|inf)"
FLAT_LINE_RE = re.compile(rf"\bstep:\d+\s+-\s+(?P<body>val-\S+:{NUMBER}(?:\s+-\s+\S+:{NUMBER})*)")
FLAT_PAIR_RE = re.compile(rf"(?P<key>val-(?:core|aux)/\S+?):(?P<val>{NUMBER})(?=\s+-\s+|\s*$)")
QUOTED_RE = re.compile(r'"([^"]*)"')


def clean(line: str) -> str:
    return RAY_PREFIX_RE.sub("", ANSI_RE.sub("", line)).rstrip("\n")


def extract_metrics(lines: list[str]) -> dict[str, float]:
    """Pull every val-core/val-aux pair out of one eval's output.

    Prefers the console logger's single flat line; falls back to stitching the pprint
    fragments only when that line is absent.
    """
    for line in reversed(lines):
        m = FLAT_LINE_RE.search(line)
        if m:
            return {p.group("key"): float(p.group("val")) for p in FLAT_PAIR_RE.finditer(m.group("body"))}

    out: dict[str, float] = {}
    buf: list[str] = []

    def flush() -> None:
        if buf:
            for m in METRIC_RE.finditer("".join(buf)):
                out[m.group("key")] = float(m.group("val"))
            buf.clear()

    for line in lines:
        quoted = QUOTED_RE.findall(line)
        if quoted:
            buf.extend(quoted)
        else:
            flush()
            for m in METRIC_RE.finditer(line):
                out[m.group("key")] = float(m.group("val"))
    flush()
    return out


def headline(metrics: dict[str, float]) -> str:
    parts = []
    for name in ("AIME24", "AIME25", "AMC23", "avg"):
        key = f"val-core/{name}/acc/mean@16"
        if key in metrics:
            parts.append(f"{name}={metrics[key]:.4f}")
    return "  ".join(parts) if parts else "(no val-core acc/mean@16 keys)"


def write_one(tag: str, out_dir: str, metrics: dict[str, float], dry_run: bool, force: bool) -> str:
    """Return one of 'written' | 'skipped' | 'pending'."""
    if not metrics:
        # Still generating, or it died before the metric dump. Nothing to write, and the
        # sweep will correctly redo it.
        print(f"[pending] {tag}: no metrics in log -> {out_dir}")
        return "pending"

    assert os.path.isdir(out_dir), f"{tag}: metrics found but output dir missing: {out_dir}"
    gen = os.path.join(out_dir, "validation", "0.jsonl")
    assert os.path.isfile(gen), f"{tag}: metrics found but generations missing at {gen}"

    nonfinite = sorted(k for k, v in metrics.items() if not math.isfinite(v))
    if nonfinite:
        print(f"[warn] {tag}: {len(nonfinite)} non-finite metric(s), e.g. {nonfinite[:3]}")

    target = os.path.join(out_dir, "validation", "metrics.json")
    if os.path.exists(target) and not force:
        print(f"[skip] {tag}: exists -> {target}")
        return "skipped"

    print(f"[{'plan' if dry_run else 'write'}] {tag}: {len(metrics)} metrics  {headline(metrics)}")
    if dry_run:
        return "skipped"

    with open(target, "w") as fh:
        json.dump({"global_step": 0, "metrics": dict(sorted(metrics.items()))}, fh, indent=2)
    return "written"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-log", help="log covering many evals, split on the sweep's markers")
    ap.add_argument("--eval-log", help="log holding exactly one eval's stdout")
    ap.add_argument("--out-dir", help="destination for --eval-log")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    assert bool(args.sweep_log) != bool(args.eval_log), "pass exactly one of --sweep-log / --eval-log"

    tally = {"written": 0, "skipped": 0, "pending": 0}

    if args.eval_log:
        assert args.out_dir, "--eval-log requires --out-dir"
        assert os.path.isfile(args.eval_log), f"eval log not found: {args.eval_log}"
        with open(args.eval_log, errors="replace") as fh:
            lines = [clean(l) for l in fh]
        tally[write_one(os.path.basename(args.out_dir), args.out_dir,
                        extract_metrics(lines), args.dry_run, args.force)] += 1
    else:
        assert os.path.isfile(args.sweep_log), f"sweep log not found: {args.sweep_log}"
        sections: list[tuple[str, str, list[str]]] = []
        with open(args.sweep_log, errors="replace") as fh:
            for raw in fh:
                line = clean(raw)
                m = MARKER_RE.search(line)
                if m:
                    sections.append((m.group("tag").strip(), m.group("out_dir"), []))
                elif sections:
                    sections[-1][2].append(line)
        assert sections, f"no '>>> ... evaluating ... -> ...' markers in {args.sweep_log}"

        for tag, out_dir, body in sections:
            tally[write_one(tag, out_dir, extract_metrics(body), args.dry_run, args.force)] += 1

    print(f"\n{tally['written']} written, {tally['skipped']} skipped, {tally['pending']} incomplete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
