#!/usr/bin/env bash
# Build SOLUTION.pdf from SOLUTION.md (pandoc -> HTML -> Edge headless print).
set -e
cd "$(dirname "$0")/.."
pandoc SOLUTION.md -s --embed-resources --css docs/doc.css -o docs/solution.html --metadata pagetitle="Promo Finder"
"/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$(cygpath -w "$PWD/SOLUTION.pdf")" "file:///$(cygpath -m "$PWD/docs/solution.html")" 2>/dev/null
python -c "import re;d=open('SOLUTION.pdf','rb').read();print('pages:',len(re.findall(rb'/Type\s*/Page[^s]',d)))"
