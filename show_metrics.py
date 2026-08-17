#!/usr/bin/env python3
"""
📊 INTERACTIVE METRICS DASHBOARD
Loads CSV and displays aggregated metrics interactively
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import time

# ============================================================================
# COLORS
# ============================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def print_header():
    print()
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}📊  CONTACT CENTER METRICS DASHBOARD  📊{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BLUE}📁 Load data from CSV and view aggregated metrics{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print()

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.GREEN}▶ {title}{Colors.END}")
    print(f"{Colors.CYAN}{'-'*60}{Colors.END}")

def print_metric(name, value, color=Colors.GREEN, unit=""):
    print(f"  {Colors.BOLD}{name}:{Colors.END} {color}{value}{Colors.END}{unit}")

# ============================================================================
# METRICS DASHBOARD
# ============================================================================
class MetricsDashboard:
    def __init__(self):
        self.df = None
        self.filepath = None
        self.data_loaded = False
    
    def load_csv(self, filepath):
        """Load CSV file."""
        if not os.path.exists(filepath):
            print(f"{Colors.RED}❌ File not found: {filepath}{Colors.END}")
            return False
        
        try:
            self.df = pd.read_csv(filepath)
            self.filepath = filepath
            self.data_loaded = True
            print(f"{Colors.GREEN}✅ Loaded {len(self.df):,} rows from {filepath}{Colors.END}")
            return True
        except Exception as e:
            print(f"{Colors.RED}❌ Error loading file: {e}{Colors.END}")
            return False
    
    def get_column(self, col_name, default=None):
        """Safely get column data."""
        if col_name in self.df.columns:
            return self.df[col_name].dropna()
        return None
    
    def scale_oh(self, value):
        """Scale OPERATIONS_HEALTH from 0-1 to 0-100 if needed."""
        if value is None:
            return None
        # If OPERATIONS_HEALTH is stored as decimal (0-1), scale to percentage (0-100)
        if isinstance(value, (int, float)):
            if value <= 1.0:
                return value * 100
        return value
    
    def show_metrics(self):
        """Display aggregated metrics."""
        if not self.data_loaded or self.df is None:
            print(f"{Colors.RED}❌ No data loaded{Colors.END}")
            return
        
        clear_screen()
        print_header()
        
        # File info
        print_section("📋 Data Overview")
        print(f"  {Colors.BOLD}File:{Colors.END} {self.filepath}")
        print(f"  {Colors.BOLD}Rows:{Colors.END} {len(self.df):,}")
        print(f"  {Colors.BOLD}Columns:{Colors.END} {len(self.df.columns)}")
        if 'date' in self.df.columns:
            print(f"  {Colors.BOLD}Date Range:{Colors.END} {self.df['date'].iloc[0]} to {self.df['date'].iloc[-1]}")
        
        # 1. Operational Health
        self.show_operational_health()
        
        # 2. KPI Metrics
        self.show_kpi_metrics()
        
        # 3. NPS Metrics
        self.show_nps_metrics()
        
        # 4. Call Volume
        self.show_call_volume()
        
        # 5. Event Distribution
        self.show_events()
        
        # 6. Summary
        self.show_summary()
        
        print()
        input(f"{Colors.DIM}Press Enter to continue...{Colors.END}")
    
    def show_operational_health(self):
        """Show Operational Health metrics."""
        print_section("🏥 Operational Health")
        
        oh_cols = ['operational_health', 'avg_score', 'avg_operational_health', 'operations_health']
        oh_data = None
        oh_name = None
        
        for col in oh_cols:
            if col in self.df.columns:
                oh_data = self.df[col].dropna()
                oh_name = col
                break
        
        if oh_data is not None and len(oh_data) > 0:
            # Scale OPERATIONS_HEALTH if needed (check if values are 0-1 range)
            if oh_data.max() <= 1.0:
                oh_data_scaled = oh_data * 100
                mean_val = oh_data_scaled.mean()
                median_val = oh_data_scaled.median()
                min_val = oh_data_scaled.min()
                max_val = oh_data_scaled.max()
            else:
                mean_val = oh_data.mean()
                median_val = oh_data.median()
                min_val = oh_data.min()
                max_val = oh_data.max()
            
            # Determine status
            if mean_val >= 92:
                status = f"{Colors.GREEN}Excellent{Colors.END}"
            elif mean_val >= 88:
                status = f"{Colors.GREEN}Healthy{Colors.END}"
            elif mean_val >= 84:
                status = f"{Colors.YELLOW}Watch{Colors.END}"
            elif mean_val >= 80:
                status = f"{Colors.YELLOW}Poor{Colors.END}"
            else:
                status = f"{Colors.RED}Critical{Colors.END}"
            
            print_metric("Mean", f"{mean_val:.2f}%")
            print_metric("Median", f"{median_val:.2f}%")
            print_metric("Min", f"{min_val:.2f}%", Colors.YELLOW)
            print_metric("Max", f"{max_val:.2f}%", Colors.GREEN)
            print_metric("Status", status)
        else:
            print(f"  {Colors.DIM}No OPERATIONS_HEALTH data available{Colors.END}")
    
    def show_kpi_metrics(self):
        """Show KPI metrics."""
        print_section("🎯 KPI Performance")
        
        kpi_mappings = [
            ('actual_quality', 'Quality', '%'),
            ('avg_quality', 'Quality', '%'),
            ('quality', 'Quality', '%'),
            ('actual_competency', 'Competency', '%'),
            ('avg_competency', 'Competency', '%'),
            ('competency', 'Competency', '%'),
            ('actual_release_rate', 'Release Rate', '%'),
            ('avg_release_rate', 'Release Rate', '%'),
            ('release_rate', 'Release Rate', '%'),
            ('actual_transfer_rate', 'Transfer Rate', '%'),
            ('avg_transfer_rate', 'Transfer Rate', '%'),
            ('transfer_rate', 'Transfer Rate', '%'),
            ('actual_attendance', 'Attendance', '%'),
            ('avg_attendance', 'Attendance', '%'),
            ('attendance', 'Attendance', '%'),
        ]
        
        found = False
        for col, name, unit in kpi_mappings:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    # Check if values need scaling
                    if data.max() <= 1.0 and name in ['Quality', 'Competency', 'Release Rate', 'Attendance']:
                        mean_val = data.mean() * 100
                        median_val = data.median() * 100
                        min_val = data.min() * 100
                        max_val = data.max() * 100
                    else:
                        mean_val = data.mean()
                        median_val = data.median()
                        min_val = data.min()
                        max_val = data.max()
                    
                    print(f"\n  {Colors.BOLD}{name}:{Colors.END}")
                    print(f"    Mean:   {mean_val:.2f}{unit}")
                    print(f"    Median: {median_val:.2f}{unit}")
                    print(f"    Min:    {min_val:.2f}{unit}")
                    print(f"    Max:    {max_val:.2f}{unit}")
                    found = True
        
        if not found:
            print(f"  {Colors.DIM}No KPI data available{Colors.END}")
    
    def show_nps_metrics(self):
        """Show NPS metrics."""
        print_section("⭐ NPS Metrics")
        
        # Direct NPS column
        if 'nps' in self.df.columns:
            data = self.df['nps'].dropna()
            if len(data) > 0:
                print(f"\n  {Colors.BOLD}NPS Score:{Colors.END}")
                print(f"    Mean:   {data.mean():.2f}")
                print(f"    Median: {data.median():.2f}")
                print(f"    Min:    {data.min():.2f}")
                print(f"    Max:    {data.max():.2f}")
        
        # Survey distribution
        if 'promoters' in self.df.columns and 'detractors' in self.df.columns:
            total_promoters = self.df['promoters'].sum()
            total_detractors = self.df['detractors'].sum()
            total_surveys = self.df['total_surveys'].sum() if 'total_surveys' in self.df.columns else 0
            
            if total_surveys > 0:
                nps = ((total_promoters - total_detractors) / total_surveys) * 100
                print(f"\n  {Colors.BOLD}Aggregate NPS:{Colors.END} {Colors.GREEN}{nps:.2f}{Colors.END}")
                print(f"\n  {Colors.BOLD}Survey Distribution:{Colors.END}")
                print(f"    {Colors.GREEN}Promoters:{Colors.END}  {total_promoters:,} ({total_promoters/total_surveys*100:.1f}%)")
                print(f"    {Colors.YELLOW}Passives:{Colors.END}   {self.df['passives'].sum():,} ({self.df['passives'].sum()/total_surveys*100:.1f}%)")
                print(f"    {Colors.RED}Detractors:{Colors.END}  {total_detractors:,} ({total_detractors/total_surveys*100:.1f}%)")
                print(f"    {Colors.BOLD}Total Surveys:{Colors.END} {total_surveys:,}")
            else:
                print(f"  {Colors.DIM}No survey data available{Colors.END}")
        else:
            print(f"  {Colors.DIM}No NPS data available{Colors.END}")
    
    def show_call_volume(self):
        """Show call volume metrics."""
        print_section("📞 Call Volume")
        
        call_cols = ['total_calls_received', 'total_calls', 'calls', 'calls_per_day']
        call_data = None
        call_name = None
        
        for col in call_cols:
            if col in self.df.columns:
                call_data = self.df[col].dropna()
                call_name = col
                break
        
        if call_data is not None and len(call_data) > 0:
            print(f"\n  {Colors.BOLD}Call Volume:{Colors.END}")
            print(f"    Mean:   {call_data.mean():.0f}")
            print(f"    Median: {call_data.median():.0f}")
            print(f"    Min:    {call_data.min():.0f}")
            print(f"    Max:    {call_data.max():.0f}")
            print(f"    Total:  {call_data.sum():,.0f}")
        else:
            print(f"  {Colors.DIM}No call volume data available{Colors.END}")
    
    def show_events(self):
        """Show event distribution."""
        print_section("🎯 Event Distribution")
        
        if 'event' in self.df.columns:
            events = self.df['event'].value_counts()
            total = len(self.df)
            
            for event, count in events.items():
                pct = (count / total) * 100
                bar = "█" * int(pct / 2)
                print(f"  {event:20} {bar} {pct:>5.1f}% ({count:,})")
        else:
            print(f"  {Colors.DIM}No event data available{Colors.END}")
    
    def show_summary(self):
        """Show summary dashboard."""
        print_section("📈 Summary Dashboard")
        
        # Collect key metrics
        oh_val = None
        quality_val = None
        competency_val = None
        release_val = None
        transfer_val = None
        nps_val = None
        
        # OPERATIONS_HEALTH
        for col in ['operational_health', 'avg_score']:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    if data.max() <= 1.0:
                        oh_val = data.mean() * 100
                    else:
                        oh_val = data.mean()
                break
        
        # Quality
        for col in ['actual_quality', 'avg_quality', 'quality']:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    if data.max() <= 1.0:
                        quality_val = data.mean() * 100
                    else:
                        quality_val = data.mean()
                break
        
        # Competency
        for col in ['actual_competency', 'avg_competency', 'competency']:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    if data.max() <= 1.0:
                        competency_val = data.mean() * 100
                    else:
                        competency_val = data.mean()
                break
        
        # Release
        for col in ['actual_release_rate', 'avg_release_rate', 'release_rate']:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    if data.max() <= 1.0:
                        release_val = data.mean() * 100
                    else:
                        release_val = data.mean()
                break
        
        # Transfer
        for col in ['actual_transfer_rate', 'avg_transfer_rate', 'transfer_rate']:
            if col in self.df.columns:
                data = self.df[col].dropna()
                if len(data) > 0:
                    transfer_val = data.mean()
                break
        
        # NPS
        if 'nps' in self.df.columns:
            nps_val = self.df['nps'].mean()
        elif 'promoters' in self.df.columns:
            total_promoters = self.df['promoters'].sum()
            total_detractors = self.df['detractors'].sum()
            total_surveys = self.df['total_surveys'].sum() if 'total_surveys' in self.df.columns else 0
            if total_surveys > 0:
                nps_val = ((total_promoters - total_detractors) / total_surveys) * 100
        
        print(f"\n  {Colors.BOLD}Key Metrics:{Colors.END}")
        if oh_val is not None:
            print(f"    OPERATIONS_HEALTH:           {oh_val:.2f}%")
        if quality_val is not None:
            print(f"    Quality:      {quality_val:.2f}%")
        if competency_val is not None:
            print(f"    Competency:   {competency_val:.2f}%")
        if release_val is not None:
            print(f"    Release Rate: {release_val:.2f}%")
        if transfer_val is not None:
            print(f"    Transfer Rate:{transfer_val:.2f}%")
        if nps_val is not None:
            print(f"    NPS:          {nps_val:.2f}")

# ============================================================================
# INTERACTIVE MENU
# ============================================================================
def interactive_menu():
    """Interactive menu for loading and viewing metrics."""
    dashboard = MetricsDashboard()
    
    while True:
        clear_screen()
        print_header()
        
        # Show current status
        if dashboard.data_loaded:
            print(f"{Colors.GREEN}✅ Data loaded: {dashboard.filepath} ({len(dashboard.df):,} rows){Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠️ No data loaded{Colors.END}")
        
        print()
        print(f"{Colors.BOLD}Main Menu{Colors.END}")
        print()
        print(f"  {Colors.YELLOW}[1]{Colors.END} Load CSV file")
        print(f"  {Colors.YELLOW}[2]{Colors.END} View Metrics")
        print(f"  {Colors.YELLOW}[3]{Colors.END} View Summary Only")
        print(f"  {Colors.YELLOW}[4]{Colors.END} Show Raw Data Preview")
        print(f"  {Colors.YELLOW}[5]{Colors.END} Save Report")
        print(f"  {Colors.YELLOW}[6]{Colors.END} Exit")
        print()
        
        choice = input(f"{Colors.GREEN}Enter choice (1-6):{Colors.END} ").strip()
        
        if choice == '1':
            clear_screen()
            print_header()
            print(f"{Colors.BLUE}📁 Load CSV File{Colors.END}")
            print()
            print("  [1] training/ai_data_full.csv")
            print("  [2] training/ai_data_large.csv")
            print("  [3] training/ai_data_medium.csv")
            print("  [4] training/ai_data_small.csv")
            print("  [5] Custom path")
            print()
            sub_choice = input(f"{Colors.GREEN}Choose (1-5):{Colors.END} ").strip()
            
            presets = {
                '1': 'training/ai_data_full.csv',
                '2': 'training/ai_data_large.csv',
                '3': 'training/ai_data_medium.csv',
                '4': 'training/ai_data_small.csv',
            }
            
            if sub_choice in presets:
                filepath = presets[sub_choice]
            elif sub_choice == '5':
                filepath = input(f"{Colors.GREEN}Enter file path:{Colors.END} ").strip()
            else:
                print(f"{Colors.RED}Invalid choice{Colors.END}")
                time.sleep(1)
                continue
            
            if dashboard.load_csv(filepath):
                print(f"{Colors.GREEN}✅ Loaded successfully!{Colors.END}")
            time.sleep(1)
        
        elif choice == '2':
            if not dashboard.data_loaded:
                print(f"{Colors.RED}❌ Please load a CSV file first{Colors.END}")
                time.sleep(1)
                continue
            dashboard.show_metrics()
        
        elif choice == '3':
            if not dashboard.data_loaded:
                print(f"{Colors.RED}❌ Please load a CSV file first{Colors.END}")
                time.sleep(1)
                continue
            clear_screen()
            print_header()
            dashboard.show_summary()
            print()
            input(f"{Colors.DIM}Press Enter to continue...{Colors.END}")
        
        elif choice == '4':
            if not dashboard.data_loaded:
                print(f"{Colors.RED}❌ Please load a CSV file first{Colors.END}")
                time.sleep(1)
                continue
            clear_screen()
            print_header()
            print_section("�� Raw Data Preview")
            print(f"\n  {Colors.DIM}First 5 rows:{Colors.END}")
            print()
            print(dashboard.df.head(5).to_string())
            print()
            print(f"  {Colors.DIM}Last 5 rows:{Colors.END}")
            print()
            print(dashboard.df.tail(5).to_string())
            print()
            input(f"{Colors.DIM}Press Enter to continue...{Colors.END}")
        
        elif choice == '5':
            if not dashboard.data_loaded:
                print(f"{Colors.RED}❌ Please load a CSV file first{Colors.END}")
                time.sleep(1)
                continue
            
            filename = input(f"{Colors.GREEN}Enter report filename (default: metrics_report.txt):{Colors.END} ").strip()
            if not filename:
                filename = "metrics_report.txt"
            
            # Save summary to file
            with open(filename, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("METRICS REPORT\n")
                f.write("=" * 80 + "\n\n")
                
                # Write summary
                f.write(f"File: {dashboard.filepath}\n")
                f.write(f"Rows: {len(dashboard.df):,}\n")
                f.write(f"Columns: {len(dashboard.df.columns)}\n\n")
                
                # Write column stats
                f.write("-" * 60 + "\n")
                f.write("COLUMN STATISTICS\n")
                f.write("-" * 60 + "\n\n")
                
                for col in dashboard.df.columns:
                    if pd.api.types.is_numeric_dtype(dashboard.df[col]):
                        data = dashboard.df[col].dropna()
                        if len(data) > 0:
                            f.write(f"{col}:\n")
                            f.write(f"  Mean:   {data.mean():.4f}\n")
                            f.write(f"  Median: {data.median():.4f}\n")
                            f.write(f"  Min:    {data.min():.4f}\n")
                            f.write(f"  Max:    {data.max():.4f}\n")
                            f.write(f"  Std:    {data.std():.4f}\n\n")
                
                f.write("-" * 60 + "\n")
                f.write("END OF REPORT\n")
                f.write("-" * 60 + "\n")
            
            print(f"{Colors.GREEN}✅ Report saved to: {filename}{Colors.END}")
            time.sleep(1)
        
        elif choice == '6':
            print(f"\n{Colors.GREEN}👋 Goodbye!{Colors.END}")
            break
        
        else:
            print(f"{Colors.RED}❌ Invalid choice{Colors.END}")
            time.sleep(1)

def main():
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️ Interrupted{Colors.END}")
        sys.exit(0)

if __name__ == "__main__":
    main()

def load_csv(filepath):
    return MetricsDashboard().load_csv(filepath)

def get_column(col_name, default=None):
    return MetricsDashboard().get_column(col_name, default)

def scale_oh(value):
    return MetricsDashboard().scale_oh(value)

def show_metrics():
    return MetricsDashboard().show_metrics()

def show_operational_health():
    return MetricsDashboard().show_operational_health()

def show_kpi_metrics():
    return MetricsDashboard().show_kpi_metrics()

def show_nps_metrics():
    return MetricsDashboard().show_nps_metrics()

def show_call_volume():
    return MetricsDashboard().show_call_volume()

def show_events():
    return MetricsDashboard().show_events()

def show_summary():
    return MetricsDashboard().show_summary()

def interactive_menu():
    return main()
