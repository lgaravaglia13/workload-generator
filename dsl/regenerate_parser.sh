# Regenerates the standalone lark parser from grammar.lark.
# parser.py is generated: never edit it by hand.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f grammar.lark ]; then
    echo "grammar.lark not found in $(pwd)" >&2
    exit 1
fi

# Generate into a temp file, then move into place: a failed run
# must not leave a truncated or empty parser.py behind.
tmp=$(mktemp parser.py.XXXXXX)
trap 'rm -f "$tmp"' EXIT
python3 -m lark.tools.standalone grammar.lark > "$tmp"
chmod 644 "$tmp"
mv "$tmp" parser.py
echo "dsl/parser.py regenerated from grammar.lark"
