#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Uso: $0 archivo.md [salida.pdf]"
  exit 1
fi

INPUT="$1"
OUTPUT="${2:-${INPUT%.md}.pdf}"

pandoc "$INPUT" -o "$OUTPUT" \
  --pdf-engine=xelatex \
  --metadata title="$(basename "$INPUT" .md)" \
  --toc \
  --highlight-style=tango

echo "Convertido: $INPUT -> $OUTPUT"