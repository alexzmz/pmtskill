"""允许执行 ``python -m src1 ...``。"""

from .pmtskill_v2.cli import main


if __name__ == "__main__":
    raise SystemExit(main())

