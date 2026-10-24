"""Command line: python -m judge {adapt,eval,baseline}. Defaults follow the organizers' container layout."""
import argparse
import asyncio

from . import pipeline


def main():
    parser = argparse.ArgumentParser(prog="judge")
    parser.add_argument("stage", choices=["adapt", "eval", "baseline"])
    parser.add_argument("--train", default="/input/train.jsonl")
    parser.add_argument("--test", default="/input/test.jsonl")
    parser.add_argument("--artifacts", default="/artifacts")
    parser.add_argument("--output", default="/output")
    args = parser.parse_args()
    if args.stage == "adapt":
        asyncio.run(pipeline.adapt(args.train, args.artifacts))
    elif args.stage == "eval":
        asyncio.run(pipeline.evaluate(args.test, args.artifacts, args.output))
    else:
        asyncio.run(pipeline.baseline(args.test, args.output))


if __name__ == "__main__":
    main()
