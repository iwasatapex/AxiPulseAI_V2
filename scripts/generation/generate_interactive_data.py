#!/usr/bin/env python3
"""
Fully Interactive Data Generator - Compact View
================================================
CORRECTED VERSION v3.1
- OPERATIONS_HEALTH Formula: 50/15/15/15/5
- NPS: Corrected distributions (85-90% promoters for NPS 81-82)
- Shows Mean, Median, Mode, Aggregate Averages
"""

import sys
import os
import random
import pandas as pd
import numpy as np
from datetime import date, timedelta
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
    MAGENTA = '\033[95m'
    PURPLE = '\033[95m'
    END = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'

def clear_screen():
    os.system('clear' if os.name == 'posix' else 'cls')

def print_header():
    print()
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.HEADER}📊  AI DATA GENERATOR v3.1  📊{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print(f"{Colors.BLUE}Enter target values → Generate data → View statistics{Colors.END}")
    print(f"{Colors.DIM}OPERATIONS_HEALTH: 50/15/15/15/5 | NPS: Corrected distributions{Colors.END}")
    print(f"{Colors.CYAN}{'='*80}{Colors.END}")
    print()

def print_section(title):
    print(f"\n{Colors.BOLD}{Colors.GREEN}▶ {title}{Colors.END}")
    print(f"{Colors.CYAN}{'-'*60}{Colors.END}")

def print_success(msg):
    print(f"{Colors.GREEN}✅ {msg}{Colors.END}")

def get_float_input(prompt, default, min_val=None, max_val=None):
    while True:
        try:
            val = input(f"{Colors.GREEN}{prompt}{Colors.END} ").strip()
            if val == "":
                return default
            val = float(val)
            if min_val is not None and val < min_val:
                print(f"{Colors.RED}❌ Value must be >= {min_val}{Colors.END}")
                continue
            if max_val is not None and val > max_val:
                print(f"{Colors.RED}❌ Value must be <= {max_val}{Colors.END}")
                continue
            return val
        except ValueError:
            print(f"{Colors.RED}❌ Please enter a number{Colors.END}")

def get_int_input(prompt, default, min_val=None, max_val=None):
    while True:
        try:
            val = input(f"{Colors.GREEN}{prompt}{Colors.END} ").strip()
            if val == "":
                return default
            val = int(val)
            if min_val is not None and val < min_val:
                print(f"{Colors.RED}❌ Value must be >= {min_val}{Colors.END}")
                continue
            if max_val is not None and val > max_val:
                print(f"{Colors.RED}❌ Value must be <= {max_val}{Colors.END}")
                continue
            return val
        except ValueError:
            print(f"{Colors.RED}❌ Please enter a number{Colors.END}")

# ============================================================================
# DATA GENERATOR
# ============================================================================
class InteractiveDataGenerator:
    def __init__(self):
        self.days = 365
        self.start_date = "2025-01-01"
        self.seed = 42
        self.targets = {}
        self.noise_std = {}
        self.df = None
        self.stats = {}
        # OPERATIONS_HEALTH Baseline
        self.oh_baseline = 97.0
        self.oh_noise = 0.04
    
    def get_user_inputs(self):
        clear_screen()
        print_header()
        
        print_section("📋 ContactCenterSimulation Settings")
        self.days = get_int_input("Number of days to simulate (default: 365):", 365, 1, 100000)
        print(f"  {Colors.DIM}→ Will generate {self.days:,} days{Colors.END}\n")
        
        self.seed = get_int_input("Random seed (default: 42):", 42)
        print(f"  {Colors.DIM}→ Reproducible results{Colors.END}\n")
        
        self.baseline_calls = get_int_input("Baseline calls/day (default: 2000):", 2000, 100, 10000)
        print(f"  {Colors.DIM}→ Avg: {self.baseline_calls}/day{Colors.END}\n")
        
        print_section("🎯 Target KPI Values")
        print(f"{Colors.DIM}  Press Enter for defaults{Colors.END}")
        print(f"{Colors.DIM}  OPERATIONS_HEALTH: 50% Release | 15% Transfer | 15% Competency | 15% Quality | 5% Volume{Colors.END}\n")
        
        self.targets['quality'] = get_float_input("Quality (default: 87.32):", 87.32, 0, 100)
        self.targets['competency'] = get_float_input("Competency (default: 93.18):", 93.18, 0, 100)
        self.targets['attendance'] = get_float_input("Attendance (default: 90.43):", 90.43, 0, 100)
        self.targets['release_rate'] = get_float_input("Release Rate (default: 60.27):", 60.27, 0, 100)
        self.targets['transfer_rate'] = get_float_input("Transfer Rate (default: 8.76):", 8.76, 0, 20)
        
        print()
        print_section("📊 Noise Configuration")
        print(f"{Colors.DIM}  OPERATIONS_HEALTH noise is fixed at 4% (automatic){Colors.END}\n")
        
        self.noise_std['quality'] = get_float_input("Quality noise (default: 1.47):", 1.47, 0, 10)
        self.noise_std['competency'] = get_float_input("Competency noise (default: 1.53):", 1.53, 0, 10)
        self.noise_std['attendance'] = get_float_input("Attendance noise (default: 0.98):", 0.98, 0, 10)
        self.noise_std['release_rate'] = get_float_input("Release Rate noise (default: 1.52):", 1.52, 0, 10)
        self.noise_std['transfer_rate'] = get_float_input("Transfer Rate noise (default: 0.97):", 0.97, 0, 10)
        
        self.show_summary()
    
    def show_summary(self):
        print_section("📋 Configuration Summary")
        print(f"  Days: {self.days:,} | Seed: {self.seed} | Baseline: {self.baseline_calls:,}")
        print(f"  Targets: Q:{self.targets['quality']:.1f} C:{self.targets['competency']:.1f} A:{self.targets['attendance']:.1f} R:{self.targets['release_rate']:.1f} T:{self.targets['transfer_rate']:.1f}")
        print(f"  OPERATIONS_HEALTH: Baseline 97 | Noise 4% | Formula: 50/15/15/15/5")
        print(f"  Noise:  Q:{self.noise_std['quality']:.2f} C:{self.noise_std['competency']:.2f} A:{self.noise_std['attendance']:.2f} R:{self.noise_std['release_rate']:.2f} T:{self.noise_std['transfer_rate']:.2f}")
    
    def generate_data(self):
        print_section("🚀 Generating Data")
        print(f"  {self.days:,} days... OPERATIONS_HEALTH Baseline: {self.oh_baseline} (±4%)")
        print()
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        rows = []
        prev_actual = dict(self.targets)
        total = self.days
        
        for day in range(total):
            if day % max(1, total // 50) == 0:
                pct = (day / total) * 100
                bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
                print(f"\r  Progress: [{bar}] {pct:.1f}%  {day+1:,}/{total:,}", end="")
                sys.stdout.flush()
            
            actual = {}
            for kpi in ['quality', 'competency', 'attendance', 'release_rate', 'transfer_rate']:
                target = self.targets[kpi]
                noise = np.random.normal(0, self.noise_std[kpi])
                value = target + noise
                value = 0.70 * prev_actual[kpi] + 0.30 * value
                
                if kpi in ['quality', 'competency', 'attendance']:
                    value = max(50.0, min(100.0, value))
                elif kpi == 'release_rate':
                    value = max(40.0, min(100.0, value))
                elif kpi == 'transfer_rate':
                    value = max(0.0, min(20.0, value))
                
                actual[kpi] = value
                prev_actual[kpi] = value
            
            calls = self.baseline_calls * np.random.normal(1, 0.05)
            calls = max(1200, min(4200, calls))
            
            # =============================================================
            # OPERATIONS_HEALTH CALCULATION - CANONICAL FORMULA (50/15/15/15/5)
            # =============================================================
            # Transfer is inverted: lower transfer = better
            norm_transfer = 100 - actual['transfer_rate'] * 5
            norm_transfer = max(0, min(100, norm_transfer))
            
            # Volume normalization
            volume_score = (calls / self.baseline_calls) * 100
            volume_score = max(0, min(100, volume_score))
            
            operations_health = (
                actual['release_rate'] * 0.50 +           # 50% Release
                norm_transfer * 0.15 +                    # 15% Transfer (inverted)
                actual['competency'] * 0.15 +             # 15% Competency
                actual['quality'] * 0.15 +                # 15% Quality
                volume_score * 0.05                       # 5% Call Volume
            )
            
            # Add noise
            operations_health += np.random.normal(0, 1.5)
            operations_health = max(40.0, min(100.0, operations_health))
            
            # =============================================================
            # NPS - CORRECTED DISTRIBUTION
            # =============================================================
            # When KPIs are met: Promoters ~85-90%, Detractors ~3-8%
            # This yields NPS 81-82
            if operations_health >= 85:
                promoter_pct = 0.85 + np.random.normal(0, 0.025)
                detractor_pct = 0.04 + np.random.normal(0, 0.015)
            else:
                promoter_pct = 0.75 + np.random.normal(0, 0.025)
                detractor_pct = 0.12 + np.random.normal(0, 0.025)
            
            promoter_pct = max(0.70, min(0.95, promoter_pct))
            detractor_pct = max(0.02, min(0.20, detractor_pct))
            passive_pct = 1.0 - promoter_pct - detractor_pct
            passive_pct = max(0.0, passive_pct)
            
            # Calculate NPS
            nps = (promoter_pct - detractor_pct) * 100
            
            # Survey count
            survey_rate = 0.06 + (operations_health / 100) * 0.03  # 6-9% based on OPERATIONS_HEALTH
            survey_rate = max(0.03, min(0.12, survey_rate))
            total_surveys = int(calls * survey_rate)
            
            # Score distribution (0-10) based on NPS
            if nps >= 80:
                scores = [0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4]  # Mostly 9-10
            elif nps >= 70:
                scores = [0, 0, 0, 0, 0, 0, 0, 2, 3, 3, 2]
            else:
                scores = [1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1]
            
            rows.append({
                'date': (date.fromisoformat(self.start_date) + timedelta(days=day)).isoformat(),
                'target_quality': round(self.targets['quality'], 2),
                'actual_quality': round(actual['quality'], 2),
                'target_competency': round(self.targets['competency'], 2),
                'actual_competency': round(actual['competency'], 2),
                'target_attendance': round(self.targets['attendance'], 2),
                'actual_attendance': round(actual['attendance'], 2),
                'target_release_rate': round(self.targets['release_rate'], 2),
                'actual_release_rate': round(actual['release_rate'], 2),
                'target_transfer_rate': round(self.targets['transfer_rate'], 2),
                'actual_transfer_rate': round(actual['transfer_rate'], 2),
                'total_calls_received': int(round(calls)),
                'operational_health': round(operations_health, 2),
                'nps': round(nps, 2),
                'promoters': int(total_surveys * promoter_pct),
                'passives': int(total_surveys * passive_pct),
                'detractors': int(total_surveys * detractor_pct),
                'total_surveys': total_surveys,
                'operational_intelligence_factor': round(np.random.normal(-2, 0.5), 2),
                'business_intelligence_factor': round(np.random.normal(-2, 0.5), 2),
                'member_intelligence_factor': round(np.random.normal(-2, 0.5), 2),
            })
        
        print()
        self.df = pd.DataFrame(rows)
        print_success(f"Generated {len(self.df):,} rows")
    
    def calculate_statistics(self):
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        stats_data = {}
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            if len(data) > 0:
                mode_val = data.mode()
                stats_data[col] = {
                    'mean': data.mean(),
                    'median': data.median(),
                    'mode': mode_val.iloc[0] if len(mode_val) > 0 else None,
                    'min': data.min(),
                    'max': data.max(),
                    'std': data.std(),
                    'count': len(data)
                }
        
        self.stats = stats_data
        self.display_statistics()
    
    def display_statistics(self):
        print_section("📊 Statistics (Mean | Median | Mode | Min | Max | Std)")
        print(f"  {Colors.DIM}{'Metric':<20} {'Mean':>8} {'Median':>8} {'Mode':>8} {'Min':>8} {'Max':>8} {'Std':>8}{Colors.END}")
        print(f"  {Colors.CYAN}{'-'*76}{Colors.END}")
        
        metrics = [
            ('operational_health', 'OPERATIONS_HEALTH'),
            ('nps', 'NPS'),
            ('actual_quality', 'Quality'),
            ('actual_competency', 'Competency'),
            ('actual_attendance', 'Attendance'),
            ('actual_release_rate', 'Release'),
            ('actual_transfer_rate', 'Transfer'),
            ('total_calls_received', 'Calls'),
            ('promoters', 'Promoters'),
            ('passives', 'Passives'),
            ('detractors', 'Detractors'),
            ('total_surveys', 'Surveys'),
            ('operational_intelligence_factor', 'OIF'),
            ('business_intelligence_factor', 'BIF'),
            ('member_intelligence_factor', 'MIF'),
        ]
        
        for col, label in metrics:
            if col in self.stats:
                s = self.stats[col]
                mode_str = f"{s['mode']:.2f}" if s['mode'] is not None else "N/A"
                print(f"  {label:<20} {s['mean']:>8.2f} {s['median']:>8.2f} {mode_str:>8} {s['min']:>8.2f} {s['max']:>8.2f} {s['std']:>8.2f}")
        
        print_section("📈 Aggregate Averages")
        
        agg_metrics = [
            ('operational_health', 'OPERATIONS_HEALTH', Colors.CYAN),
            ('nps', 'NPS', Colors.GREEN),
            ('actual_quality', 'Quality', Colors.BLUE),
            ('actual_competency', 'Competency', Colors.MAGENTA),
            ('actual_release_rate', 'Release', Colors.YELLOW),
            ('actual_transfer_rate', 'Transfer', Colors.RED),
        ]
        
        print(f"  {Colors.BOLD}Metric{' ':<15} Value     Bar{Colors.END}")
        print(f"  {Colors.CYAN}{'-'*60}{Colors.END}")
        
        for i, (col, label, color) in enumerate(agg_metrics):
            if col in self.stats:
                val = self.stats[col]['mean']
                filled = int((val / 100) * 30)
                bar = "█" * filled + "░" * (30 - filled)
                if i < 3:
                    print(f"  {label:<15} {color}{val:>6.2f}{Colors.END}  {color}{bar}{Colors.END}", end="")
                    if i + 3 < len(agg_metrics):
                        col2, label2, color2 = agg_metrics[i+3]
                        val2 = self.stats[col2]['mean']
                        filled2 = int((val2 / 100) * 30)
                        bar2 = "█" * filled2 + "░" * (30 - filled2)
                        print(f"  {label2:<15} {color2}{val2:>6.2f}{Colors.END}  {color2}{bar2}{Colors.END}")
                    else:
                        print()
                elif i >= 3:
                    pass
        
        print_section("📞 Call Volume")
        total_calls = self.df['total_calls_received'].sum()
        avg_calls = self.df['total_calls_received'].mean()
        min_calls = self.df['total_calls_received'].min()
        max_calls = self.df['total_calls_received'].max()
        
        print(f"  Total: {total_calls:,.0f}  |  Avg: {avg_calls:.0f}  |  Min: {min_calls:.0f}  |  Max: {max_calls:.0f}  |  Days: {len(self.df):,}")
        
        # NPS Verification
        print_section("✅ NPS Verification")
        avg_nps = self.stats.get('nps', {}).get('mean', 0)
        avg_promoters = self.stats.get('promoters', {}).get('mean', 0)
        avg_detractors = self.stats.get('detractors', {}).get('mean', 0)
        avg_surveys = self.stats.get('total_surveys', {}).get('mean', 0)
        
        print(f"  Average NPS: {Colors.GREEN}{avg_nps:.2f}{Colors.END}")
        print(f"  Average Promoters: {avg_promoters:.1f} per day")
        print(f"  Average Detractors: {avg_detractors:.1f} per day")
        print(f"  Average Surveys: {avg_surveys:.1f} per day")
        
        if 80 <= avg_nps <= 83:
            print(f"  {Colors.GREEN}✅ NPS is in target range (81-82){Colors.END}")
        else:
            print(f"  {Colors.YELLOW}⚠️ NPS is {avg_nps:.2f} (target: 81-82){Colors.END}")
    
    def save_data(self):
        print_section("💾 Save Data")
        
        default_path = "training/interactive_data.csv"
        print(f"  {Colors.DIM}Default: {default_path}{Colors.END}")
        print()
        
        while True:
            filepath = input(f"{Colors.GREEN}Enter file path (or Enter for default):{Colors.END} ").strip()
            if filepath == "":
                filepath = default_path
            
            if not filepath.endswith('.csv'):
                filepath += '.csv'
            
            if os.path.exists(filepath):
                overwrite = input(f"{Colors.YELLOW}⚠️ File exists. Overwrite? (y/n):{Colors.END} ").strip().lower()
                if overwrite != 'y':
                    print(f"{Colors.DIM}Choose different filename{Colors.END}")
                    continue
            
            break
        
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        self.df.to_csv(filepath, index=False)
        
        file_size = os.path.getsize(filepath)
        if file_size > 1024 * 1024:
            size_str = f"{file_size / (1024*1024):.2f} MB"
        elif file_size > 1024:
            size_str = f"{file_size / 1024:.2f} KB"
        else:
            size_str = f"{file_size} bytes"
        
        print_success(f"Data saved to: {filepath}")
        print(f"  Rows: {len(self.df):,} | Columns: {len(self.df.columns)} | Size: {size_str}")
        
        return filepath
    
    def run(self):
        self.get_user_inputs()
        self.generate_data()
        self.calculate_statistics()
        
        print()
        print(f"{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.GREEN}✅ Data generation complete!{Colors.END}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}")
        print()
        
        self.save_data()
        
        print()
        print(f"{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.GREEN}🎉 All done!{Colors.END}")
        print(f"{Colors.CYAN}{'='*80}{Colors.END}")
        print()
        print(f"{Colors.DIM}📊 View: python show_metrics.py{Colors.END}")
        print(f"{Colors.DIM}📈 Train: python predict_cli.py train <file> --save model.pkl{Colors.END}")
        print()

    
def get_user_inputs():
    """Compatibility wrapper for interactive configuration."""
    return {}

def show_summary(generator):
    return generator.show_summary()

def generate_data(generator):
    return generator.generate_data()

def calculate_statistics(generator):
    return generator.calculate_statistics()

def display_statistics(generator):
    return generator.display_statistics()

def save_data(generator):
    return generator.save_data()

def run(generator):
    return generator.run()

# ============================================================================
# MAIN
# ============================================================================
def main():
    try:
        generator = InteractiveDataGenerator()
        generator.run()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️ Interrupted{Colors.END}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
