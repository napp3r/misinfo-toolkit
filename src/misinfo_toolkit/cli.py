"""Command line interface: ``misinfo <command> [options]``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from . import RANDOM_SEED, __version__
from .data import Splits, load_dataset, load_mide22_tsv, stratified_split
from .evaluate import results_table, run_experiment, write_report
from .models import MODELS, build_model, fake_probability, load_model, save_model


def _cmd_prepare(args: argparse.Namespace) -> None:
    df = load_mide22_tsv(args.input)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df[["text", "label"]].to_csv(args.output, index=False)
    counts = df["label"].value_counts().to_dict()
    print(f"Saved {len(df)} tweets to {args.output} (label counts: {counts})")


def _cmd_split(args: argparse.Namespace) -> None:
    splits = stratified_split(
        load_dataset(args.input), val_size=args.val_size, test_size=args.test_size, seed=args.seed
    )
    splits.save(args.output)
    print(json.dumps(splits.stats(), indent=2))


def _cmd_evaluate(args: argparse.Namespace) -> None:
    results = run_experiment(Splits.load(args.splits), args.models, seed=args.seed)
    write_report(results, args.output)
    print(results_table(results).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nReport written to {args.output}/")


def _cmd_train(args: argparse.Namespace) -> None:
    df = load_dataset(args.input)
    model = build_model(args.model, seed=args.seed)
    model.fit(df["text"], df["label"])
    save_model(model, args.output)
    print(f"Trained '{args.model}' on {len(df)} tweets -> {args.output}")


def _cmd_predict(args: argparse.Namespace) -> None:
    model = load_model(args.model)
    for text, p in zip(args.text, fake_probability(model, args.text), strict=True):
        verdict = "MISINFORMATION" if p >= 0.5 else "truthful"
        print(f"{p:.3f}\t{verdict}\t{text}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="misinfo", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare", help="binarise a hydrated MiDe22 TSV into text,label CSV")
    p.add_argument("input", help="path to mide22_en_misinfo_tweets.tsv (with a 'text' column)")
    p.add_argument("-o", "--output", default="data/processed/dataset.csv")
    p.set_defaults(func=_cmd_prepare)

    p = sub.add_parser("split", help="stratified train/val/test split")
    p.add_argument("input", help="text,label CSV")
    p.add_argument("-o", "--output", default="data/splits")
    p.add_argument("--val-size", type=float, default=0.10)
    p.add_argument("--test-size", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.set_defaults(func=_cmd_split)

    p = sub.add_parser("evaluate", help="train/evaluate models and write a report")
    p.add_argument("splits", help="directory with train.csv, val.csv, test.csv")
    p.add_argument("-m", "--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    p.add_argument("-o", "--output", default="reports")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.set_defaults(func=_cmd_evaluate)

    p = sub.add_parser("train", help="fit one model on a CSV and save it with joblib")
    p.add_argument("input", help="text,label CSV")
    p.add_argument("-m", "--model", choices=list(MODELS), default="logreg")
    p.add_argument("-o", "--output", default="models/model.joblib")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("predict", help="score texts with a saved model")
    p.add_argument("text", nargs="+")
    p.add_argument("--model", default="models/model.joblib")
    p.set_defaults(func=_cmd_predict)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
