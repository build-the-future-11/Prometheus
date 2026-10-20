from sparking.cli import build_parser


def test_cli_exposes_full_scientific_lifecycle():
    help_text = build_parser().format_help()
    for command in ("prepare", "preflight", "develop", "confirm", "verify"):
        assert command in help_text

