"""
Report Generation (JSON, CSV, HTML)
"""
import json
import csv
from datetime import datetime
from pathlib import Path

class ReportGenerator:
    def __init__(self, analytics_results=None):
        self.results = analytics_results or {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def add_results(self, name, data):
        self.results[name] = data
        return self
    
    def to_json(self, filepath=None):
        if filepath is None:
            filepath = f"analytics_report_{self.timestamp}.json"
        
        with open(filepath, 'w') as f:
            json.dump({
                "generated_at": datetime.now().isoformat(),
                "results": self.results
            }, f, indent=2, default=str)
        return filepath
    
    def to_csv(self, filepath=None):
        if filepath is None:
            filepath = f"analytics_report_{self.timestamp}.csv"
        
        # Flatten nested results for CSV
        flat_data = []
        for category, values in self.results.items():
            if isinstance(values, dict):
                for key, val in values.items():
                    flat_data.append({"category": category, "metric": key, "value": val})
            elif isinstance(values, list):
                for item in values:
                    if isinstance(item, dict):
                        item["category"] = category
                        flat_data.append(item)
        
        with open(filepath, 'w', newline='') as f:
            if flat_data:
                writer = csv.DictWriter(f, fieldnames=flat_data[0].keys())
                writer.writeheader()
                writer.writerows(flat_data)
        return filepath
    
    def to_html(self, filepath=None):
        if filepath is None:
            filepath = f"analytics_report_{self.timestamp}.html"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AxiPulseAI Analytics Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; margin-top: 25px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .section {{ background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                .timestamp {{ color: #7f8c8d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <h1>📊 AxiPulseAI Analytics Report</h1>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        for category, data in self.results.items():
            if isinstance(data, dict):
                html += f'<div class="section"><h2>{category.replace("_", " ").title()}</h2><table>'
                for key, val in data.items():
                    if isinstance(val, (int, float, str)):
                        html += f"<tr><td>{key}</td><td>{val}</td></tr>"
                html += "</table></div>"
        
        html += """
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html)
        return filepath
    
    def print_summary(self):
        """Print a human-readable summary."""
        print("\n" + "="*70)
        print("📊 AxiPulseAI Analytics Summary")
        print("="*70)
        
        for category, data in self.results.items():
            if isinstance(data, dict):
                print(f"\n📌 {category.replace('_', ' ').title()}")
                print("-"*40)
                for key, val in data.items():
                    if isinstance(val, (int, float)):
                        if isinstance(val, float):
                            print(f"  {key.replace('_', ' ').title():20} : {val:.2f}")
                        else:
                            print(f"  {key.replace('_', ' ').title():20} : {val:,}")
                    elif isinstance(val, str):
                        print(f"  {key.replace('_', ' ').title():20} : {val}")
        print("\n" + "="*70)

def generate_full_report(data_path, model_paths=None, output_prefix='analytics_report'):
    """
    Generate a complete analytics report from a data file.
    """
    from .data_analytics import DataAnalytics
    from .business_analytics import BusinessAnalytics
    
    report = ReportGenerator()
    
    # Data Quality
    da = DataAnalytics(data_path=data_path)
    report.add_results("data_quality", da.quality_report())
    report.add_results("outliers", da.outlier_detection())
    report.add_results("summary_stats", da.summary_stats())
    
    # Business Metrics
    ba = BusinessAnalytics(data_path=data_path)
    report.add_results("nps_summary", ba.nps_summary())
    report.add_results("oh_summary", ba.oh_summary())
    report.add_results("kpi_gaps", ba.kpi_gap_analysis())
    report.add_results("call_patterns", ba.call_patterns())
    report.add_results("release_transfer", ba.release_transfer_analysis())
    
    # Save reports
    json_file = report.to_json(f"{output_prefix}.json")
    csv_file = report.to_csv(f"{output_prefix}.csv")
    html_file = report.to_html(f"{output_prefix}.html")
    
    report.print_summary()
    print(f"\n📁 Reports saved to:")
    print(f"  JSON: {json_file}")
    print(f"  CSV : {csv_file}")
    print(f"  HTML: {html_file}")
    
    return report

# Module-level compatibility surface

def add_results(name, data):
    return ReportGenerator().add_results(name, data)

def to_json(filepath=None):
    return ReportGenerator().to_json(filepath)

def to_csv(filepath=None):
    return ReportGenerator().to_csv(filepath)

def to_html(filepath=None):
    return ReportGenerator().to_html(filepath)

def print_summary():
    return ReportGenerator().print_summary()
