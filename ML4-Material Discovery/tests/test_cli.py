from ml4materials.cli import build_parser


def test_cli_exposes_all_required_interfaces() -> None:
    parser = build_parser()
    for command in ("prepare", "run", "predict", "rank-csv", "verify", "manifest"):
        args = [command]
        if command == "run":
            args += ["--run-id", "fixture"]
        elif command == "predict":
            args += ["--checkpoint", "model.joblib", "--formula", "GaAs"]
        elif command == "rank-csv":
            args += ["--checkpoint", "model.joblib", "--input", "in.csv", "--output", "out.csv"]
        elif command in {"verify", "manifest"}:
            args += ["--run-id", "fixture"]
        assert parser.parse_args(args).command == command
