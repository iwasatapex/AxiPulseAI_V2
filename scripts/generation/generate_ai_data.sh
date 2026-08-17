#!/bin/bash
# Quick launcher for AI training data generation

echo "🤖 AI Training Data Generator (NO NPS SCORES)"
echo "=============================================="
echo ""
echo "⚠️  NOTE: NPS scores are NOT included in the output."
echo "   The AI models will learn to predict NPS from the features."
echo ""
echo "Choose dataset size:"
echo "  1) Quick (30 days)"
echo "  2) Small (100 days)"
echo "  3) Medium (365 days)"
echo "  4) Large (1000 days)"
echo "  5) Full (10000 days)"
echo "  6) Custom"
echo ""
read -p "Enter choice (1-6): " choice

case $choice in
    1)
        DAYS=30
        OUTPUT="training/ai_data_quick.csv"
        ;;
    2)
        DAYS=100
        OUTPUT="training/ai_data_small.csv"
        ;;
    3)
        DAYS=365
        OUTPUT="training/ai_data_medium.csv"
        ;;
    4)
        DAYS=1000
        OUTPUT="training/ai_data_large.csv"
        ;;
    5)
        DAYS=10000
        OUTPUT="training/ai_data_full.csv"
        ;;
    6)
        read -p "Enter number of days: " DAYS
        read -p "Enter output filename: " OUTPUT
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

read -p "Enter number of agents (default: 10): " AGENTS
AGENTS=${AGENTS:-10}

echo ""
echo "Select season (case-insensitive):"
echo "  NORMAL - Default operations"
echo "  AEP - Annual Enrollment Period (+80% volume)"
echo "  OEP - Open Enrollment Period (+45% volume)"
echo "  BENEFIT_ACTIVATION - January benefit activation"
echo "  RANDOM - Mix of all seasons"
echo ""
read -p "Enter season (default: NORMAL): " SEASON_INPUT
SEASON_INPUT=${SEASON_INPUT:-NORMAL}

# Convert to uppercase for validation
SEASON=$(echo "$SEASON_INPUT" | tr '[:lower:]' '[:upper:]')

# Validate season
case $SEASON in
    NORMAL|AEP|OEP|BENEFIT_ACTIVATION|RANDOM)
        ;;
    *)
        echo "⚠️  Invalid season '$SEASON_INPUT'. Using NORMAL."
        SEASON="NORMAL"
        ;;
esac

echo ""
echo "Generating training data..."
echo "  Days: $DAYS"
echo "  Agents: $AGENTS"
echo "  Season: $SEASON"
echo "  Output: $OUTPUT"
echo "  NPS Score: NOT included (models learn to predict)"
echo ""

python3 generate_training_data.py --days $DAYS --agents $AGENTS --season $SEASON --output $OUTPUT

echo ""
echo "✅ Data generation complete!"
echo "📁 File: $OUTPUT"
echo "📊 This file contains RAW survey data (promoters/passives/detractors/scores)"
echo "   AI models will learn to predict NPS from operational metrics"
