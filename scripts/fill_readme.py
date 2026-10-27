"""Fill README.md's results block from results/heldout/summary.json.   python scripts/fill_readme.py"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docextract import report  # noqa: E402

res = json.loads((ROOT / "results" / "heldout" / "summary.json").read_text(encoding="utf-8"))
body = report.markdown(res)
readme = ROOT / "README.md"
text = readme.read_text(encoding="utf-8")
a, b = "<!-- results:start -->", "<!-- results:end -->"
i, j = text.find(a), text.find(b)
if i == -1 or j < i:
    raise SystemExit("README markers not found")
readme.write_text(text[: i + len(a)] + "\n" + body + "\n" + text[j:], encoding="utf-8")
print("updated", readme)
