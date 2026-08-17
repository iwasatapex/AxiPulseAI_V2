#!/bin/bash
# Cleanup redundant files in AxiPulseAI project

echo "🧹 Cleaning up redundant files..."
echo ""

# Files to remove (redundant/duplicate)
REDUNDANT_FILES=(
    # Duplicate/old simulator files
    "run.py"
    "run_sim.py"
    "simulation.py"
    "simulator.py"
    "run_simulator_interactive.py"
    "agent.py"
    
    # Old/generated test files
    "debug_generate.py"
    "generate_aggregate_data.py"
    "generate_training_data.py"
    "test_config_import.py"
    "validate_configs.py"
    "update_kpi_imports.py"
    
    # Old backup files
    "core/simulator/config.py.bak"
    "core/simulator/kpi.py.bak"
    
    # Old OH engine v2 (replaced)
    "core/simulator/oh_engine_v2.py"
)

echo "Removing redundant files..."
for file in "${REDUNDANT_FILES[@]}"; do
    if [ -f "$file" ]; then
        rm -f "$file"
        echo "  ✅ Removed: $file"
    else
        echo "  ⏭️  Not found: $file"
    fi
done

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📁 Files kept in root:"
ls -la *.py 2>/dev/null | awk '{print "  " $9}' | grep -v "__init__"

echo ""
echo "📁 Core simulator files:"
ls -la core/simulator/*.py 2>/dev/null | head -5

