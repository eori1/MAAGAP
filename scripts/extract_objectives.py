import re
import sys

import docx


def main(path: str) -> int:
    d = docx.Document(path)
    lines = [p.text.strip() for p in d.paragraphs if p.text and p.text.strip()]

    pat = re.compile(
        r"(objective\s*3|objective\s*4|specific\s+objectives|research\s+objectives)",
        re.IGNORECASE,
    )
    idx = [i for i, t in enumerate(lines) if pat.search(t)]
    print(f"matches={len(idx)}")

    for i in idx:
        start = max(0, i - 6)
        end = min(len(lines), i + 45)
        print(f"\n--- context starting line {i} ---")
        for j in range(start, end):
            sys.stdout.buffer.write((f"{j}: {lines[j]}\n").encode("utf-8", "replace"))

    return 0


if __name__ == "__main__":
    docx_path = sys.argv[1] if len(sys.argv) > 1 else "UG-CICT-THESIS-MANUSCRIPT_LATEST_UPDATE.docx"
    raise SystemExit(main(docx_path))

