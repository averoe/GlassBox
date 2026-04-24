"""Bulk type annotation modernization script.

Replaces Dict->dict, List->list, Tuple->tuple, Type->type,
Optional[X] -> X | None across all source files that have
'from __future__ import annotations'.
"""

import os
import re

src_dir = os.path.join("src", "glassbox_rag")
files = []
for root, dirs, fnames in os.walk(src_dir):
    dirs[:] = [d for d in dirs if d != "__pycache__"]
    for f in fnames:
        if f.endswith(".py"):
            files.append(os.path.join(root, f))

count = 0
for fpath in files:
    with open(fpath, "r", encoding="utf-8") as f:
        content = f.read()

    original = content

    if "from __future__ import annotations" not in content:
        continue

    content = content.replace("Dict[", "dict[")
    content = content.replace("List[", "list[")
    content = content.replace("Tuple[", "tuple[")
    content = content.replace("Type[", "type[")

    # Optional[simple_type] -> simple_type | None
    content = re.sub(r"Optional\[(\w+)\]", r"\1 | None", content)
    # Optional[dict[...]] -> dict[...] | None
    content = re.sub(r"Optional\[(dict\[[^\]]+\])\]", r"\1 | None", content)
    content = re.sub(r"Optional\[(list\[[^\]]+\])\]", r"\1 | None", content)

    if content != original:
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(content)
        count += 1
        print(f"Updated: {fpath}")

print(f"\nTotal files updated: {count}")
