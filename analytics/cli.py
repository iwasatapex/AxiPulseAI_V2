"""
Analytics CLI – Run from terminal
"""
import argparse
from .reporting import generate_full_report
from .model_analytics import ModelAnalytics

def main():
    parser = argparse.ArgumentParser(description="AxiPulseAI Analytics CLI")
    parser.add_argument("--data", type=str, required=True, help="Path to data CSV file")
    parser.add_argument("--model", type=str, help="Path to model PKL file (optional)")
    parser.add_argument("--output", type=str, default="analytics_report", help="Output file prefix")
    args = parser.parse_args()
    
    # Generate report
    generate_full_report(args.data, output_prefix=args.output)
    
    # If model is provided, evaluate it
    if args.model:
        print("\n🔍 Evaluating model on data...")
        ma = ModelAnalytics(model_path=args.model, data_path=args.data)
        # Try to infer target columns
        df = ma.df
        if 'operational_health' in df.columns:
            X_cols = [c for c in df.columns if c not in ['operational_health', 'date']]
            y_cols = ['operational_health']
            results = ma.evaluate_on_data(X_cols, y_cols)
            print(f"Model Performance:")
            for k, v in results.items():
                print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
