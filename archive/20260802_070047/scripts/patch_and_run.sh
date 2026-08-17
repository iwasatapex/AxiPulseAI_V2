#!/usr/bin/env bash
set -Eeuo pipefail

trap 'echo "❌ FAILED at line $LINENO: $BASH_COMMAND" >&2' ERR

# ------------------------------------------------------------------
# Project directory
# ------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

echo "============================================================"
echo " Contact Center Simulator – V2.4 Patch & Run"
echo "============================================================"
echo "Project : $PROJECT_DIR"

# ------------------------------------------------------------------
# Locate Python virtual environment
# ------------------------------------------------------------------
if [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/venv/bin/python"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    echo "❌ No working virtual environment found."
    echo "Expected one of:"
    echo "  $PROJECT_DIR/venv/bin/python"
    echo "  $PROJECT_DIR/.venv/bin/python"
    exit 1
fi

echo "Python  : $PYTHON"
"$PYTHON" --version
echo "============================================================"

SIM_DIR="$PROJECT_DIR/core/contact_center_simulator"
if [[ ! -d "$SIM_DIR" ]]; then
    echo "❌ Simulator directory not found: $SIM_DIR"
    exit 1
fi

# ------------------------------------------------------------------
# 1. Fix survey.py syntax error (idempotent)
# ------------------------------------------------------------------
echo "🔧 Fixing survey.py syntax..."
SURVEY_FILE="$SIM_DIR/survey.py"
if [[ -f "$SURVEY_FILE" ]]; then
    # Remove any line that starts with 'def     ' (malformed)
    sed -i '/^def     /d' "$SURVEY_FILE"
fi

# ------------------------------------------------------------------
# 2. Apply all patches using Python (idempotent)
# ------------------------------------------------------------------
echo "📦 Applying V2.4 patches (idempotent)..."

"$PYTHON" << 'PYEOF'
import re, os, sys
from pathlib import Path

SIM_DIR = Path("core/contact_center_simulator")
if not SIM_DIR.exists():
    print("❌ Simulator directory not found")
    sys.exit(1)

# ----------------------------------------------------------------
# 2a. Patch simulation.py: add get_training_data and modify run_day
# ----------------------------------------------------------------
SIM_FILE = SIM_DIR / "simulation.py"
if not SIM_FILE.exists():
    print("❌ simulation.py not found")
    sys.exit(1)

with open(SIM_FILE, 'r') as f:
    content = f.read()

# ---- Add get_training_data if missing ----
if "def get_training_data(self)" not in content:
    # Insert before the last method or at the end of the class
    class_pattern = r'(class ContactCenterSimulator.*?:)(.*?)(?=\nclass |\Z)'
    match = re.search(class_pattern, content, re.DOTALL)
    if match:
        class_body = match.group(2)
        new_method = '''
    def get_training_data(self):
        """Return list of dicts for training.csv."""
        training_rows = []
        for day in self.history:
            row = {
                'date': day.get('date'),
                'operational_health': day.get('avg_operational_health', 0),
                'business_intelligence_factor': day.get('avg_business_intel_factor', 0),
                'member_intelligence_factor': day.get('avg_member_intel_factor', 0),
                'target_release_rate': day.get('target_release_rate', 60.0),
                'actual_release_rate': day.get('effective_release_rate', 0),
                'total_calls_received': day.get('calls', 0),
            }
            score_counts = day.get('score_counts_total', [0]*11)
            for i in range(11):
                row[f'score_{i}'] = int(score_counts[i]) if i < len(score_counts) else 0
            training_rows.append(row)
        return training_rows
'''
        class_body = class_body.rstrip() + '\n' + new_method + '\n'
        content = content.replace(match.group(2), class_body)
        print("✅ Inserted get_training_data")
    else:
        print("⚠️ Could not find class ContactCenterSimulator")

# ---- Modify run_day to compute new fields if missing ----
if "avg_business_intel_factor" not in content:
    lines = content.split('\n')
    new_lines = []
    inserted = False
    for i, line in enumerate(lines):
        if not inserted and line.strip().startswith('day_summary = {'):
            insert_code = [
                '',
                '        # --- Training data fields ---',
                '        avg_business_intel_factor = np.mean([(a.business_intelligence - 82.5) / 12.5 * 100 for a in self.agents])',
                '        avg_member_intel_factor = np.mean([(a.member_intelligence - 82.5) / 12.5 * 100 for a in self.agents])',
                '        score_counts_total = [0]*11',
                '        for ar in agent_results:',
                '            nps_res = ar.get("nps_result", {})',
                '            sc = nps_res.get("score_counts", [0]*11)',
                '            for j in range(11):',
                '                score_counts_total[j] += sc[j] if j < len(sc) else 0',
                '        effective_release_rate = (total_released / total_calls_handled * 100) if total_calls_handled > 0 else 0.0',
                '        target_release_rate = get_targets().release if hasattr(get_targets(), "release") else 60.0',
                '',
            ]
            new_lines.extend(insert_code)
            inserted = True
        new_lines.append(line)

    if inserted:
        content = '\n'.join(new_lines)
        # Add fields to day_summary dict
        pattern = r'(day_summary = \{)(.*?)(\})'
        def add_fields(m):
            header = m.group(1)
            body = m.group(2)
            closing = m.group(3)
            new_fields = '''
            "avg_business_intel_factor": avg_business_intel_factor,
            "avg_member_intel_factor": avg_member_intel_factor,
            "score_counts_total": score_counts_total,
            "effective_release_rate": effective_release_rate,
            "target_release_rate": target_release_rate,
'''
            body = body.rstrip() + ',\n' + new_fields
            return header + body + closing
        content = re.sub(pattern, add_fields, content, flags=re.DOTALL)
        print("✅ Modified run_day with training fields")
    else:
        print("⚠️ Could not find day_summary line")

# ---- Ensure get_targets is imported ----
if "get_targets" not in content:
    # Add to the import list
    content = content.replace('from .kpi import (', 'from .kpi import (\n    get_targets,')
    print("✅ Added get_targets import")

with open(SIM_FILE, 'w') as f:
    f.write(content)

# ----------------------------------------------------------------
# 2b. Patch run.py to write training.csv
# ----------------------------------------------------------------
RUN_FILE = SIM_DIR / "run.py"
if not RUN_FILE.exists():
    print("❌ run.py not found")
    sys.exit(1)

with open(RUN_FILE, 'r') as f:
    content = f.read()

# ---- Add import csv if missing ----
if "import csv" not in content:
    content = "import csv\n" + content

# ---- Add training.csv writing after default CSV ----
if "training.csv" not in content:
    pattern = r'(with open\(output_path, \'w\', newline=\'\'\) as f:.*?writer\.writerows\(results\))'
    def repl(m):
        before = m.group(0)
        new_code = before + '''

        # --- Write training.csv ---
        training_data = sim.get_training_data()
        if training_data:
            training_path = f"training/training.csv"
            os.makedirs(os.path.dirname(training_path), exist_ok=True)
            with open(training_path, 'w', newline='') as tf:
                fieldnames = [
                    'date', 'operational_health', 'business_intelligence_factor',
                    'member_intelligence_factor', 'target_release_rate', 'actual_release_rate',
                    'total_calls_received',
                    'score_0', 'score_1', 'score_2', 'score_3', 'score_4',
                    'score_5', 'score_6', 'score_7', 'score_8', 'score_9', 'score_10'
                ]
                writer = csv.DictWriter(tf, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(training_data)
            print(f"✅ Training data written to: {training_path}")
'''
        return new_code
    content = re.sub(pattern, repl, content, flags=re.DOTALL | re.MULTILINE)
    print("✅ Added training.csv writing to run.py")

with open(RUN_FILE, 'w') as f:
    f.write(content)

print("✅ All Python patches applied successfully.")
PYEOF

# ------------------------------------------------------------------
# 3. Ensure operations_health_engine.py is not empty (stub)
# ------------------------------------------------------------------
OH_FILE="$SIM_DIR/operations_health_engine.py"
if [[ ! -f "$OH_FILE" ]]; then
    echo "Creating stub operations_health_engine.py"
    cat > "$OH_FILE" << 'EOF'
# Stub implementation for operations_health_engine
class OHEngine:
    def run(self, *args, **kwargs):
        return OHResult()

class OHState:
    pass

class KPIActuals:
    def __init__(self, quality=87.0, competency=93.0, attendance=90.0, release=60.0, transfer=9.0):
        self.quality = quality
        self.competency = competency
        self.attendance = attendance
        self.release = release
        self.transfer = transfer

class OHResult:
    def __init__(self, score=80.0):
        self.score = score

# Aliases for compatibility
OperationsHealthEngine = OHEngine
OperationsHealthState = OHState
OperationsHealthResult = OHResult
EOF
fi

# ------------------------------------------------------------------
# 4. Run the simulation
# ------------------------------------------------------------------
echo "============================================================"
echo "🚀 Starting simulation..."
echo "============================================================"

cd "$PROJECT_DIR"
"$PYTHON" "$PROJECT_DIR/core/contact_center_simulator/run.py"
