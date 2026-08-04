"""
`dense-evolution` console script (see pyproject.toml [project.scripts]).

Only one real subcommand today: `serve`, which starts the local Composer
kernel (local_site.app.server) that the published Composer page
(docs/composer.md) talks to. fastapi/uvicorn/pydantic are the
`dense-evolution[composer]` extra, not core dependencies -- imported here,
inside the serve branch, not at module level, so `import dense_evolution`
itself never requires them.
"""

import sys

USAGE = """usage: dense-evolution <command>

commands:
  serve    Start the local Composer kernel (http://127.0.0.1:8800) that
           the published Composer page (docs/composer.md) connects to.
           Requires the composer extra: pip install dense-evolution[composer]
"""


def main():
    args = sys.argv[1:]
    if args == ["serve"]:
        try:
            from local_site.app.server import main as serve_main
        except ImportError as exc:
            print(
                "dense-evolution serve needs the composer extra:\n"
                "  pip install dense-evolution[composer]\n"
                f"(missing: {exc.name})",
                file=sys.stderr,
            )
            sys.exit(1)
        serve_main()
        return

    print(USAGE, file=sys.stderr if args else sys.stdout)
    sys.exit(1 if args else 0)


if __name__ == "__main__":
    main()
