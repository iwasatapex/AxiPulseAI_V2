#!/bin/bash
# apply_v2.2_fixed.sh – corrected path handling

set -e  # exit on error

echo "=== Contact Center Simulator - V2.2 Update (fixed paths) ==="
echo "Working directory: $(pwd)"

# ----------------------------------------------------------------------
# 0. Set simulator directory
# ----------------------------------------------------------------------
SIM_DIR="core/contact_center_simulator"
if [ ! -d "$SIM_DIR" ]; then
    echo "❌ Error: $SIM_DIR not found. Are you in the project root?"
    exit 1
fi

# ----------------------------------------------------------------------
# 1. Rename all 'oh' (case variants) to 'operations_health' (run from root)
#    This will catch every .py file, including those in subdirectories.
#    We'll re-run it idempotently (safe to run multiple times).
# ----------------------------------------------------------------------
echo "Renaming oh -> operations_health in all .py files ..."
find . -type f -name "*.py" -exec sed -i.bak \
    -e 's/\boh\b/operations_health/g' \
    -e 's/\bOH\b/OPERATIONS_HEALTH/g' \
    -e 's/\bOh\b/OperationsHealth/g' \
    -e 's/\boh_score\b/operations_health_score/g' \
    -e 's/\boh_factor\b/operations_health_factor/g' \
    -e 's/\boh_effect\b/operations_health_effect/g' \
    -e 's/\b_oh\b/_operations_health/g' \
    {} +

# ----------------------------------------------------------------------
# 2. Rename files oh_engine.py -> operations_health_engine.py, etc.
# ----------------------------------------------------------------------
echo "Renaming files and updating imports ..."
cd "$SIM_DIR"
if [ -f "oh_engine.py" ]; then
    mv oh_engine.py operations_health_engine.py
fi
if [ -f "oh_bridge.py" ]; then
    mv oh_bridge.py operations_health_bridge.py
fi
cd - > /dev/null

# Update imports in all .py files (from the root)
find . -type f -name "*.py" -exec sed -i \
    -e 's/from \.oh_engine import/from .operations_health_engine import/g' \
    -e 's/from \.oh_bridge import/from .operations_health_bridge import/g' \
    -e 's/import oh_engine/import operations_health_engine/g' \
    -e 's/import oh_bridge/import operations_health_bridge/g' \
    {} +

# ----------------------------------------------------------------------
# 3. Apply V2.2 rule changes – all file edits are done inside $SIM_DIR
# ----------------------------------------------------------------------
echo "Applying V2.2 rule changes ..."
cd "$SIM_DIR"

# ----------------------------------------------------------------------
# 3a. Agent experience factors (§3.4) – replace the whole method
# ----------------------------------------------------------------------
echo "  - Agent experience factors (agent.py)"
# Remove the old function and insert the new one using awk
awk '
/def calculate_learning_multiplier\(self\):/ {
    print "    def calculate_learning_multiplier(self):"
    print "        if self.experience_weeks < 4:"
    print "            return 0.95"
    print "        elif self.experience_weeks < 8:"
    print "            return 0.97"
    print "        elif self.experience_weeks < 16:"
    print "            return 0.99"
    print "        else:"
    print "            return 1.0"
    # skip old body until next def
    while (getline > 0) {
        if (/^    def /) { print; break }
    }
    next
}
{ print }
' agent.py > agent.py.tmp && mv agent.py.tmp agent.py

# ----------------------------------------------------------------------
# 3b. kpi.py: OH_effect additive, call volume factor, OH momentum, etc.
# ----------------------------------------------------------------------
echo "  - kpi.py: OH_effect additive, call volume, momentum"

# Replace "* operations_health_effect" with "+ operations_health_effect" in release/quality formulas
sed -i '/def _base_release/,/return release/ s/\* operations_health_effect/\+ operations_health_effect/g' kpi.py
sed -i '/def _base_quality/,/return quality/ s/\* operations_health_effect/\+ operations_health_effect/g' kpi.py

# Replace _normalized_call_volume_factor with V2.2 version
sed -i '/def _normalized_call_volume_factor/,/return/ c\
    def _normalized_call_volume_factor(self, calls, baseline=2000):\
        ratio = calls / baseline\
        if ratio <= 1.0:\
            return np.clip(0.9 + 0.1 * ratio, 0.9, 1.0)\
        else:\
            return np.clip(1.0 - (ratio - 1.0) * 0.5, 0.5, 1.0)' kpi.py

# Add OH momentum to _calculate_base_oh and adjust generate_day for OH_effect application
# We'll use a Python script for multiline changes (safer than sed for complex blocks)
python3 << 'EOF'
import re, sys

# Read kpi.py
with open('kpi.py', 'r') as f:
    content = f.read()

# 1) In _calculate_base_oh, add momentum using prev.get('operations_health')
pattern = r'(def _calculate_base_oh\(.*?\):)(.*?)(?=\n    def |\Z)'
def repl_calc(m):
    head = m.group(1)
    body = m.group(2)
    # Replace the line that computes base_oh with a new version that includes previous OH
    new_body = re.sub(
        r'(base_oh = .*?)(?=\n)',
        r'base_oh = \1\n        previous_oh = prev.get("operations_health", base_oh)\n        oh_score = 0.6 * previous_oh + 0.4 * base_oh + rng.normal(0, 2.5)\n        return oh_score',
        body,
        flags=re.DOTALL
    )
    # Remove old 'return base_oh' if present
    new_body = re.sub(r'return base_oh', '', new_body)
    return head + new_body
content = re.sub(pattern, repl_calc, content, flags=re.DOTALL)

# 2) In generate_day, insert OH_effect computation and apply to release/transfer/quality
# We'll find the place where oh_score is assigned and insert the OH_effect block.
# Then we'll replace the existing assignment lines for release, transfer, quality.
pattern = r'(oh_score = .*?)(?=\n\s+# Pass 2:)'
def repl_gen(m):
    before = m.group(1)
    insertion = '''
    # OH_effect per §9.8
    oh_deviation = (oh_score - 85) / 100.0
    release_OH_effect = oh_deviation * 5.0
    transfer_OH_effect = oh_deviation * 2.0
    quality_OH_effect = oh_deviation * 3.0
    '''
    return before + insertion
content = re.sub(pattern, repl_gen, content, flags=re.DOTALL)

# Now replace the assignments to release, transfer, quality to use the OH_effect
content = re.sub(
    r'release = self\._apply_oh_effect_to_release\(base_release, oh_score\)',
    r'release = base_release + release_OH_effect',
    content
)
content = re.sub(
    r'transfer = self\._apply_oh_effect_to_transfer\(base_transfer, oh_score\)',
    r'transfer = base_transfer - transfer_OH_effect',
    content
)
content = re.sub(
    r'quality = float\(np\.clip\(base_quality, 60, 100\)\)',
    r'quality = base_quality + quality_OH_effect\n        quality = float(np.clip(quality, 60, 100))',
    content
)

with open('kpi.py', 'w') as f:
    f.write(content)
EOF

# ----------------------------------------------------------------------
# 3c. survey.py: update caps (detractors ≤5%, passives ≤6% of released calls)
# ----------------------------------------------------------------------
echo "  - survey.py: update hard caps"
sed -i 's/detractor_max_percentage = 0\.08/detractor_max_percentage = 0.05/g' survey.py
sed -i 's/passive_max_percentage = 0\.10/passive_max_percentage = 0.06/g' survey.py
# Update the validation function
sed -i '/def validate_hard_constraints/,/^def / {
    s/detractors <= 0\.08 \* released_calls/detractors <= 0.05 * released_calls/g
    s/passives <= 0\.10 \* released_calls/passives <= 0.06 * released_calls/g
}' survey.py
# Also any other occurrences
sed -i 's/0\.08 \* released_calls/0.05 * released_calls/g' survey.py
sed -i 's/0\.10 \* released_calls/0.06 * released_calls/g' survey.py

# ----------------------------------------------------------------------
# 3d. simulation.py: Critical complexity only on severe events
# ----------------------------------------------------------------------
echo "  - simulation.py: Critical complexity conditional"
python3 << 'EOF'
import re, sys
with open('simulation.py', 'r') as f:
    content = f.read()

# Find the line where complexity is chosen and replace with conditional logic
pattern = r'(complexity = self\.rng\.choice\(\[[^\]]*\], p=\[[^\]]*\]\))'
def repl(m):
    return '''
        # Determine if severe event
        severe_events = (Event.CORE_OUTAGE, Event.CMS_CHANGE)
        if event_enum in severe_events:
            # Include Critical with small probability, renormalize others
            probs = [0.30, 0.30, 0.20, 0.10, 0.05]  # Low, Medium, High, Very High, Critical
            # Normalize to sum to 1
            total = sum(probs)
            probs = [p/total for p in probs]
            complexity = self.rng.choice(['Low','Medium','High','Very High','Critical'], p=probs)
        else:
            complexity = self.rng.choice(['Low','Medium','High','Very High'], p=[0.30, 0.30, 0.20, 0.10])'''
content = re.sub(pattern, repl, content, flags=re.DOTALL)

with open('simulation.py', 'w') as f:
    f.write(content)
EOF

# ----------------------------------------------------------------------
# 4. Cleanup – remove .bak files (optional)
# ----------------------------------------------------------------------
cd - > /dev/null
echo "Removing backup files (.bak) from all directories ..."
find . -type f -name "*.bak" -delete

echo "✅ All V2.2 updates applied successfully."
echo ""
echo "Next steps:"
echo "  - Review changes in $SIM_DIR (especially kpi.py, survey.py, simulation.py)."
echo "  - Run a quick test: python run.py"
echo "  - If you encounter any NameError, ensure all 'oh' references are updated."
echo "  - The script renamed files and updated imports; check for any remaining old references."
