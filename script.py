import json
from pathlib import Path
with open(Path("./coverage.json"), mode="rt", encoding="utf-8") as f:
    config = json.load(f)
    breakpoint()
