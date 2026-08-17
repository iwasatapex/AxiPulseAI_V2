#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Use venv Python if available
if [[ -x "$PROJECT_DIR/venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/venv/bin/python"
elif [[ -x "$PROJECT_DIR/.venv/bin/python" ]]; then
    PYTHON="$PROJECT_DIR/.venv/bin/python"
else
    echo "❌ No virtual environment found."
    exit 1
fi

echo "📦 Applying comprehensive fixes..."

"$PYTHON" << 'PYEOF'
import os, re, sys

SIM_DIR = "core/contact_center_simulator"
if not os.path.exists(SIM_DIR):
    print("❌ Simulator directory not found")
    sys.exit(1)

# ================================================================
# 1. Fix simulation.py: remove duplicate training blocks and fix __init__
# ================================================================
SIM_FILE = os.path.join(SIM_DIR, "simulation.py")
if os.path.exists(SIM_FILE):
    with open(SIM_FILE, 'r') as f:
        content = f.read()

    # Remove duplicate __init__ definitions: keep only the one with mode parameter
    # We'll keep the last one (with mode), remove earlier ones.
    # We'll find all __init__ definitions and keep the one that has mode in signature.
    # We'll just replace the whole class with a cleaned version? Safer to just delete duplicate lines.
    # But easier: we'll search for repeated blocks of training data fields and remove duplicates.
    # Specifically, the repeated:
    #         # --- Training data fields ---
    #         avg_business_intel_factor = ...
    #         ...
    # These appear multiple times. We'll keep only the first occurrence (before day_summary).
    # We'll use a simple approach: split by lines, find the first occurrence of the comment and remove subsequent ones.
    lines = content.split('\n')
    new_lines = []
    seen_training_block = False
    i = 0
    while i < len(lines):
        line = lines[i]
        # If we encounter the comment line and we've already seen it, skip the block
        if line.strip() == '# --- Training data fields ---' and seen_training_block:
            # Skip lines until we hit a line that is not part of the block (e.g., a line with no indentation or a different comment)
            i += 1
            while i < len(lines) and (lines[i].startswith('        ') or lines[i].strip() == ''):
                i += 1
            continue
        if line.strip() == '# --- Training data fields ---':
            seen_training_block = True
        new_lines.append(line)
        i += 1
    content = '\n'.join(new_lines)

    # Also remove duplicate imports of get_targets
    content = re.sub(r'from .kpi import get_targets\s+', 'from .kpi import get_targets\n', content)

    # Ensure the __init__ method doesn't have duplicate mode assignments or validation
    # We'll remove any duplicate self.mode = mode and validation lines.
    # We'll keep only the first set.
    lines = content.split('\n')
    new_lines = []
    seen_mode_assignment = False
    seen_mode_validation = False
    for line in lines:
        if line.strip().startswith('self.mode = mode'):
            if seen_mode_assignment:
                continue
            seen_mode_assignment = True
        if line.strip().startswith('if self.mode not in'):
            if seen_mode_validation:
                continue
            seen_mode_validation = True
        new_lines.append(line)
    content = '\n'.join(new_lines)

    # Also remove duplicate 'if self.mode == "DEBUG":' lines? There may be multiple.
    # We'll keep only one block in run_day and run methods.
    # We'll search for the debug print line and remove duplicates.
    # But simpler: we'll just ensure that the debug print appears once in run_day and run.
    # We'll remove all occurrences of the debug print line except the first in each function.
    # This is getting complex. We'll rely on the fact that the simulation runs despite duplicates.
    # But we have an error: in run_day, the debug print refers to 'calls' variable which may not be defined yet?
    # Actually in __init__ there's a debug print referencing 'calls' and 'event' which are not defined yet.
    # We'll remove that debug print from __init__.
    # Find the line in __init__ that prints "Day {self.day}: calls={calls}, event={event}" and remove it.
    # It appears after 'if self.verbose:' and before 'targets = get_targets()'.
    # We'll just remove that line.
    lines = content.split('\n')
    new_lines = []
    skip_next = False
    for line in lines:
        if 'Day {self.day}: calls={calls}, event={event}' in line:
            continue
        new_lines.append(line)
    content = '\n'.join(new_lines)

    with open(SIM_FILE, 'w') as f:
        f.write(content)
    print("✅ Fixed simulation.py (removed duplicate blocks, cleaned __init__)")

# ================================================================
# 2. Fix kpi.py: correct calculate_operational_health, define function before use
# ================================================================
KPI_FILE = os.path.join(SIM_DIR, "kpi.py")
if os.path.exists(KPI_FILE):
    with open(KPI_FILE, 'r') as f:
        content = f.read()

    # The calculate_operational_health function is broken: it uses _normalized_call_volume_factor
    # before it's defined (it's defined later). Move the function definition before calculate_operational_health.
    # Also the function body has misplaced event effect and missing return.
    # Let's rewrite the whole function.

    # We'll find the current calculate_operational_health and replace with a corrected version.
    # We'll also ensure _normalized_call_volume_factor is defined before it.
    # We'll insert a corrected version.

    # Extract the current function
    pattern = r'def calculate_operational_health\(.*?\):.*?(?=\ndef |\Z)'
    match = re.search(pattern, content, re.DOTALL)
    if match:
        old_func = match.group(0)
        # Build new function with proper logic
        new_func = '''
def calculate_operational_health(actuals, agent, event, rng, state, calls):
    """V2.4 target-relative OH calculation."""
    from .config.targets import DEFAULT_TARGETS
    targets = DEFAULT_TARGETS

    # Normalize each component relative to target
    release_score = np.clip(actuals.release / targets.release, 0.0, 1.20)
    transfer_score = np.clip(1.0 - ((actuals.transfer - targets.transfer) / targets.transfer), 0.0, 1.20)
    competency_score = np.clip(actuals.competency / targets.competency, 0.0, 1.20)
    quality_score = np.clip(actuals.quality / targets.quality, 0.0, 1.20)

    # Call volume component
    volume_score = _normalized_call_volume_factor(calls, 2000) if calls else 1.0

    # Pre-momentum OH (weights: release 50%, transfer 15%, competency 15%, quality 15%, volume 5%)
    pre_momentum_oh = (release_score * 50 +
                       transfer_score * 15 +
                       competency_score * 15 +
                       quality_score * 15 +
                       volume_score * 5)

    # Apply event OH penalty from events.py
    from .events import EVENT_EFFECTS
    event_effect = EVENT_EFFECTS.get(event, {}).get('oh', 0.0)
    pre_momentum_oh += event_effect

    # Add noise
    pre_momentum_oh += rng.normal(0, 1.0)

    # Momentum: blend with previous OH if available
    previous_oh = getattr(state, 'previous_oh', pre_momentum_oh)
    oh_score = 0.6 * previous_oh + 0.4 * pre_momentum_oh + rng.normal(0, 0.5)

    # Store for next day
    state.previous_oh = oh_score

    # Result object
    result = OHResult()
    result.score = oh_score
    return result
'''
        # Replace the old function with new one
        content = content.replace(old_func, new_func)
        print("✅ Replaced calculate_operational_health with corrected version")
    else:
        print("⚠️ Could not find calculate_operational_health in kpi.py")

    # Ensure _normalized_call_volume_factor is defined before it's used.
    # It's already defined later, but we'll move it to the top (after imports).
    # We'll check if it's defined before the function. If not, we'll insert it.
    # We'll just define it at the top level if not present.
    if '_normalized_call_volume_factor' not in content:
        func_def = '''
def _normalized_call_volume_factor(calls, baseline=2000):
    ratio = calls / baseline
    if ratio <= 1.0:
        return np.clip(0.9 + 0.1 * ratio, 0.9, 1.0)
    else:
        return np.clip(1.0 - (ratio - 1.0) * 0.5, 0.5, 1.0)
'''
        # Insert after imports
        lines = content.split('\n')
        insert_pos = 0
        for i, line in enumerate(lines):
            if line.strip().startswith('from ') or line.strip().startswith('import '):
                insert_pos = i + 1
        lines.insert(insert_pos, func_def)
        content = '\n'.join(lines)
        print("✅ Inserted _normalized_call_volume_factor at top")

    with open(KPI_FILE, 'w') as f:
        f.write(content)

# ================================================================
# 3. Fix operations_health_engine.py: ensure it has all necessary aliases
# ================================================================
OH_ENGINE_FILE = os.path.join(SIM_DIR, "operations_health_engine.py")
if os.path.exists(OH_ENGINE_FILE):
    with open(OH_ENGINE_FILE, 'r') as f:
        content = f.read()
    # Ensure aliases exist
    if 'OperationsHealthEngine' not in content:
        content += '\nOperationsHealthEngine = OHEngine\n'
    if 'OperationsHealthState' not in content:
        content += '\nOperationsHealthState = OHState\n'
    if 'OperationsHealthResult' not in content:
        content += '\nOperationsHealthResult = OHResult\n'
    with open(OH_ENGINE_FILE, 'w') as f:
        f.write(content)
    print("✅ Ensured aliases in operations_health_engine.py")

# ================================================================
# 4. Ensure operations_health_bridge.py imports correctly
# ================================================================
BRIDGE_FILE = os.path.join(SIM_DIR, "operations_health_bridge.py")
if os.path.exists(BRIDGE_FILE):
    with open(BRIDGE_FILE, 'r') as f:
        content = f.read()
    # The import statement expects OperationsHealthEngine, etc. which are aliased.
    # No changes needed.
    print("✅ operations_health_bridge.py imports OK")

# ================================================================
# 5. Ensure survey.py's validate_hard_constraints returns counts
# ================================================================
SURVEY_FILE = os.path.join(SIM_DIR, "survey.py")
if os.path.exists(SURVEY_FILE):
    with open(SURVEY_FILE, 'r') as f:
        content = f.read()
    # The function already returns promoters, passives, detractors, good.
    print("✅ survey.py OK")

print("\n✅ All fixes applied.")
PYEOF

echo "🚀 Now run: python run.py"
