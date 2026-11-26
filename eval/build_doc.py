"""Fill eval/experiments_template.md with the generated tables -> results/final/EXPERIMENTS.md.

    python -m eval.build_doc

Markers:  {{TABLE:<name>}}  inserts results/final/tables/<name>.md (table + its source file)
          {{FILE:<path>}}   inserts a generated Markdown file (headings demoted by two levels)
Prose numbers are written by hand in the template; each cites the file it comes from.
"""
import re
from pathlib import Path

TEMPLATE = Path(__file__).with_name("experiments_template.md")
OUT = Path("results/final/EXPERIMENTS.md")
TAB = Path("results/final/tables")


def table(name: str) -> str:
    text = (TAB / f"{name}.md").read_text().split("\n")
    src = next(l for l in text if l.startswith("Source:"))
    body = [l for l in text if l.startswith("|")]
    return "\n".join(body) + f"\n\n*{src.replace('Source:', 'Source:').strip()} · LaTeX: `{TAB}/{name}.tex`*"


def file(path: str) -> str:
    text = Path(path).read_text()
    return re.sub(r"^(#+) ", lambda m: "#" * min(6, len(m.group(1)) + 2) + " ", text, flags=re.M)


def main():
    s = TEMPLATE.read_text()
    s = re.sub(r"\{\{TABLE:([\w.\-]+)\}\}", lambda m: table(m.group(1)), s)
    s = re.sub(r"\{\{FILE:([\w./\-]+)\}\}", lambda m: file(m.group(1)), s)
    assert "{{" not in s, "unfilled marker"
    OUT.write_text(s)
    print(f"wrote {OUT} ({len(s.splitlines())} lines)")


if __name__ == "__main__":
    main()
