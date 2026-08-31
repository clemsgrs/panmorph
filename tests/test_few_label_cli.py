from pathlib import Path

from experiments.run_few_label import build_parser


def test_few_label_command_uses_clear_default_result_directory() -> None:
    args = build_parser(Path("/repo")).parse_args([])

    assert args.out == Path("/repo/results/few-label")
