"""
`dense-evolution` console script (see pyproject.toml [project.scripts]).

Two real subcommands:
  serve             starts the local Composer kernel (local_site.app.server)
                    that the published Composer page (docs/composer.md)
                    talks to.
  offline-composer  downloads the real published Composer page (the same
                    HTML Github Pages serves, not a hand-rolled copy) plus
                    the same-origin assets it references, into a local
                    folder -- so it opens via file:// with no internet at
                    all, while still talking to the local kernel above.

fastapi/uvicorn/pydantic are the `dense-evolution[composer]` extra, not
core dependencies -- imported here, inside the serve branch, not at
module level, so `import dense_evolution` itself never requires them.
"""

import sys

USAGE = """usage: dense-evolution <command>

commands:
  serve                    Start the local Composer kernel (http://127.0.0.1:8800)
                           that the published Composer page (docs/composer.md)
                           connects to.
  offline-composer [DEST]  Download the real published Composer page and its
                           assets into DEST (default: ./composer-offline) so
                           it works via file:// with no internet.

Requires the composer extra: pip install dense-evolution[composer]
"""

COMPOSER_PAGE_URL = "https://tatopenn-cell.github.io/Dense-Evolution/composer/"


def _require_composer_extra():
    try:
        import fastapi, uvicorn, pydantic  # noqa: F401
    except ImportError as exc:
        print(
            "dense-evolution needs the composer extra:\n"
            "  pip install dense-evolution[composer]\n"
            f"(missing: {exc.name})",
            file=sys.stderr,
        )
        sys.exit(1)


def _cmd_serve():
    _require_composer_extra()
    from local_site.app.server import main as serve_main
    serve_main()


def _cmd_offline_composer(dest: str):
    """Mirrors COMPOSER_PAGE_URL into `dest`: the page itself plus every
    same-origin <link href>/<script src> it references (Material's shared
    theme CSS/JS bundle, the Composer-specific app.js/style.css), each
    saved at the same relative path it already uses on the live site --
    mkdocs/Material already generate those as page-relative, precisely so
    a subtree like this stays self-consistent once copied elsewhere. Not a
    full recursive mirror (fonts referenced only from inside a CSS url()
    are not followed -- Material falls back to system fonts without them,
    a cosmetic gap, not a functional one): scoped to what the page's own
    <head>/<body> actually link to, which is everything the Composer UI
    itself needs to run."""
    import os
    import urllib.request
    from html.parser import HTMLParser
    from urllib.parse import urljoin, urlparse

    class _AssetFinder(HTMLParser):
        def __init__(self):
            super().__init__()
            self.assets = []

        def handle_starttag(self, tag, attrs):
            attrs = dict(attrs)
            if tag == "link" and attrs.get("href"):
                self.assets.append(attrs["href"])
            elif tag == "script" and attrs.get("src"):
                self.assets.append(attrs["src"])
            elif tag == "img" and attrs.get("src"):
                self.assets.append(attrs["src"])

    print(f"Scarico {COMPOSER_PAGE_URL} ...")
    with urllib.request.urlopen(COMPOSER_PAGE_URL, timeout=30) as resp:
        html_bytes = resp.read()
    html_text = html_bytes.decode("utf-8", errors="replace")

    parser = _AssetFinder()
    parser.feed(html_text)

    page_url = urlparse(COMPOSER_PAGE_URL)
    base_origin = f"{page_url.scheme}://{page_url.netloc}"
    # mkdocs generates every relative link (../assets/...) relative to the
    # SITE root, not to whatever folder happens to hold index.html -- so
    # the page itself has to be saved at the same depth under `dest` it
    # already has under the site root (site_root_path stripped from its
    # own path leaves "composer/"), or its own "../assets/..." references
    # end up pointing one directory above `dest` instead of inside it.
    # site_root_path is derived from COMPOSER_PAGE_URL itself (this
    # project's one fixed, known URL), not guessed from each asset URL.
    site_root_path = page_url.path.rsplit("composer/", 1)[0]

    def _relative_to_site_root(url: str) -> str:
        path = urlparse(url).path
        if path.startswith(site_root_path):
            path = path[len(site_root_path):]
        return path.lstrip("/")

    os.makedirs(dest, exist_ok=True)
    page_relative_path = _relative_to_site_root(COMPOSER_PAGE_URL) + "index.html"
    page_local_path = os.path.join(dest, page_relative_path)
    os.makedirs(os.path.dirname(page_local_path), exist_ok=True)
    with open(page_local_path, "w", encoding="utf-8") as f:
        f.write(html_text)

    seen = set()
    for href in parser.assets:
        absolute_url = urljoin(COMPOSER_PAGE_URL, href)
        if not absolute_url.startswith(base_origin) or absolute_url in seen:
            continue  # a genuinely external resource (e.g. a CDN) -- not ours to mirror
        seen.add(absolute_url)

        relative_path = _relative_to_site_root(absolute_url)
        local_path = os.path.join(dest, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            with urllib.request.urlopen(absolute_url, timeout=30) as resp:
                data = resp.read()
            with open(local_path, "wb") as f:
                f.write(data)
            print(f"  scaricato: {relative_path}")
        except Exception as exc:
            print(f"  saltato (non essenziale): {relative_path} ({exc})")

    print(f"\nCopia offline pronta: {page_local_path}")
    return page_local_path


def main():
    args = sys.argv[1:]
    if args == ["serve"]:
        _cmd_serve()
        return
    if args and args[0] == "offline-composer":
        dest = args[1] if len(args) > 1 else "composer-offline"
        _cmd_offline_composer(dest)
        return

    print(USAGE, file=sys.stderr if args else sys.stdout)
    sys.exit(1 if args else 0)


if __name__ == "__main__":
    main()
