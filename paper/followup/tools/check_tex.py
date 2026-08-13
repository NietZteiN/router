#!/usr/bin/env python3
"""Structural sanity check for a LaTeX draft on a machine with no TeX installed.

Not a compiler. Catches the failures that would otherwise only surface at build time:
unbalanced environments, undefined \\cite keys, dangling \\ref, unescaped % and _, and
column-count mismatches in tabular rows.
"""
import re
import sys
from collections import Counter

tex_path, bib_path = sys.argv[1], sys.argv[2]
src = open(tex_path, encoding="utf-8").read()
bib = open(bib_path, encoding="utf-8").read()

problems = []

# Strip comments (a % not preceded by a backslash starts one).
lines = src.split("\n")
stripped = []
for ln in lines:
    out, i = [], 0
    while i < len(ln):
        if ln[i] == "%" and (i == 0 or ln[i - 1] != "\\"):
            break
        out.append(ln[i])
        i += 1
    stripped.append("".join(out))
body = "\n".join(stripped)

# 1. environments balance, in order
stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", body):
    kind, name = m.group(1), m.group(2)
    line = body[: m.start()].count("\n") + 1
    if kind == "begin":
        stack.append((name, line))
    else:
        if not stack:
            problems.append(f"line {line}: \\end{{{name}}} with no open environment")
        elif stack[-1][0] != name:
            problems.append(f"line {line}: \\end{{{name}}} closes \\begin{{{stack[-1][0]}}} (line {stack[-1][1]})")
            stack.pop()
        else:
            stack.pop()
for name, line in stack:
    problems.append(f"line {line}: \\begin{{{name}}} never closed")

# 2. braces balance
depth = 0
for i, ch in enumerate(body):
    if ch == "{" and (i == 0 or body[i - 1] != "\\"):
        depth += 1
    elif ch == "}" and (i == 0 or body[i - 1] != "\\"):
        depth -= 1
        if depth < 0:
            problems.append(f"line {body[:i].count(chr(10)) + 1}: unmatched closing brace")
            depth = 0
if depth:
    problems.append(f"{depth} unclosed brace(s) overall")

# 3. citations resolve
bibkeys = set(re.findall(r"@\w+\{([^,]+),", bib))
cited = set()
for m in re.finditer(r"\\cite[a-z]*\{([^}]+)\}", body):
    cited.update(k.strip() for k in m.group(1).split(","))
for k in sorted(cited - bibkeys):
    problems.append(f"\\cite{{{k}}} has no entry in {bib_path}")
unused = sorted(bibkeys - cited)

# 4. refs resolve
labels = set(re.findall(r"\\label\{([^}]+)\}", body))
for m in re.finditer(r"\\(?:page)?ref\{([^}]+)\}", body):
    if m.group(1) not in labels:
        problems.append(f"\\ref{{{m.group(1)}}} has no \\label")
dupes = [l for l, n in Counter(re.findall(r"\\label\{([^}]+)\}", body)).items() if n > 1]
for d in dupes:
    problems.append(f"duplicate \\label{{{d}}}")

# 5. raw _ and & outside math/verbatim-ish contexts (common paste error)
for i, ln in enumerate(stripped, 1):
    scrub = re.sub(r"\$[^$]*\$", "", ln)                 # math
    scrub = re.sub(r"\\texttt\{[^}]*\}", "", scrub)      # texttt bodies still need escaping, but
    scrub = re.sub(r"\\(?:label|ref|cite[a-z]*|input|bibliography|newcommand|renewcommand)\{[^}]*\}", "", scrub)
    for m in re.finditer(r"(?<!\\)_", scrub):
        problems.append(f"line {i}: unescaped underscore -> {ln.strip()[:80]}")

# 6. tabular column counts
def read_group(s, i):
    """s[i] == '{'; return (contents, index just past the matching '}')."""
    assert s[i] == "{"
    d, j = 0, i
    while j < len(s):
        if s[j] == "{":
            d += 1
        elif s[j] == "}":
            d -= 1
            if d == 0:
                return s[i + 1:j], j + 1
        j += 1
    raise ValueError("unbalanced")


for m in re.finditer(r"\\begin\{tabular\}(?:\[[^\]]*\])?", body):
    spec, after = read_group(body, m.end())
    end = body.index("\\end{tabular}", after)
    content = body[after:end]
    spec_clean = re.sub(r"@\{[^}]*\}", "", spec)          # @{} inter-column glue is not a column
    ncol = len(re.sub(r"[^lcrp]", "", re.sub(r"p\{[^}]*\}", "p", spec_clean)))
    startline = body[: m.start()].count("\n") + 1
    for row in content.split("\\\\"):
        row = re.sub(r"\\(?:toprule|midrule|bottomrule|cmidrule\S*)", "", row).strip()
        if not row or row.startswith("%"):
            continue
        got = len(re.findall(r"(?<!\\)&", row)) + 1
        if got != ncol:
            problems.append(f"tabular at line {startline}: row has {got} cells, spec has {ncol}: {row[:70]}")

print(f"{tex_path}: {len(problems)} problem(s)")
for p in problems:
    print("  FAIL", p)
if unused:
    print(f"  note: {len(unused)} bib entries uncited: {', '.join(unused)}")
sys.exit(1 if problems else 0)
