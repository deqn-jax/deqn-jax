#!/usr/bin/env bash
# Build the public site into _pages/ and, with --publish, push it to GitHub Pages.
#
#   _pages/        the economist pages, copied from docs/econ/
#   _pages/docs/   the engineering site, built by mkdocs from docs/site/
#
# Pages that the engineering site used to serve at the root (before it moved
# to /docs/) get a small redirect page at their old address.
#
# Publishing goes where `mkdocs gh-deploy --remote-name pages` used to go:
# branch gh-pages of the `pages` remote (github.com/deqn-jax/deqn-jax.github.io),
# as a new commit on top of what is there, never a force push.
#
# Usage:
#   scripts/dev/build_pages.sh             # build only; preview with
#                                          #   python -m http.server -d _pages
#   scripts/dev/build_pages.sh --publish   # build and publish
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
out=_pages

rm -rf "$out"
uv run --extra docs mkdocs build --clean --site-dir "$out/docs"
cp -R docs/econ/. "$out/"
cp "$out/docs/404.html" "$out/404.html"

# Redirects from the old root addresses of the engineering pages.
(cd "$out/docs" && find . -mindepth 2 -name index.html) | while read -r page; do
  rel="${page#./}"
  rel="${rel%index.html}"
  [ -e "$out/${rel}index.html" ] && continue
  mkdir -p "$out/$rel"
  cat >"$out/${rel}index.html" <<EOF
<!doctype html>
<meta charset="utf-8">
<title>Moved</title>
<link rel="canonical" href="/docs/$rel">
<meta http-equiv="refresh" content="0; url=/docs/$rel">
<p>This page moved to <a href="/docs/$rel">/docs/$rel</a>.</p>
EOF
done

echo "built $out/ ($(find "$out" -type f | wc -l | tr -d ' ') files)"

if [ "${1:-}" = "--publish" ]; then
  sha=$(git rev-parse --short HEAD)
  git fetch pages gh-pages
  uv run --extra docs ghp-import --no-jekyll --push --remote pages \
    --branch gh-pages --message "Deploy site from $sha" "$out"
fi
