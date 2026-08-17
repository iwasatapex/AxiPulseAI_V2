from pathlib import Path
import re

TARGET = (
    Path(__file__).resolve().parents[1]
    / "generation"
    / "generate_interactive_data.py"
)

content = TARGET.read_text()

if "MAGENTA" not in content:
    content = re.sub(
        r"RED = '\\033\[91m'",
        "RED = '\\033[91m'\n    MAGENTA = '\\033[95m'",
        content,
        count=1,
    )
    TARGET.write_text(content)

print("✅ Added MAGENTA to Colors class")
