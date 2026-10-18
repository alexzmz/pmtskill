"""Eight-method entry; preserves the original five-method batch defaults."""
import sys
from .batch_evaluate import ALL_METHODS, build_parser, run


def main(argv=None):
    parser = build_parser()
    parser.description = "Eight AndroidWorld evaluation entries; prepare frozen inputs, then run serially. See EIGHT_METHOD_EVALUATION.md."
    parser.set_defaults(methods=ALL_METHODS)
    try:
        return run(parser.parse_args(argv))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Comparison failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
