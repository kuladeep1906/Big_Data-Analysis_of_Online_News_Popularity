#!/usr/bin/env python3
"""
Validation script to verify all Report 1 requirements are met.

This script checks:
- Presence of required files
- Subset file has exactly 100 rows
- Script runs successfully
- All 5 figures are generated
- Metrics are computed correctly
"""

import os
import sys
import subprocess
import pandas as pd


def check_file_exists(filepath, description):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} NOT FOUND")
        return False


def validate_project():
    """Run all validation checks."""
    print("=" * 70)
    print("Report 1 - Project Validation")
    print("=" * 70)
    print()
    
    all_checks_passed = True
    
    # Check 1: Required files exist
    print("1. Checking required files...")
    files_to_check = [
        ("data/subset.csv", "Subset file"),
        ("report_1.py", "Main analysis script"),
        ("requirements.txt", "Requirements file"),
        ("README.md", "README documentation"),
    ]
    
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_checks_passed = False
    print()
    
    # Check 2: Subset has exactly 100 rows
    print("2. Checking subset file size...")
    if os.path.exists("data/subset.csv"):
        try:
            df = pd.read_csv("data/subset.csv")
            if df.shape[0] == 100:
                print(f"✅ Subset has exactly 100 rows")
                print(f"   Columns: {df.shape[1]}")
            else:
                print(f"❌ Subset has {df.shape[0]} rows (expected 100)")
                all_checks_passed = False
        except Exception as e:
            print(f"❌ Error reading subset: {e}")
            all_checks_passed = False
    print()
    
    # Check 3: Script runs successfully
    print("3. Running report_1.py...")
    if os.path.exists("report_1.py") and os.path.exists("data/subset.csv"):
        try:
            result = subprocess.run(
                ["python", "report_1.py", "data/subset.csv"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                print("✅ Script executed successfully")
                
                # Check if "ANALYSIS" is in output
                if "ANALYSIS" in result.stdout:
                    print("✅ ANALYSIS section found in output")
                else:
                    print("❌ ANALYSIS section not found in output")
                    all_checks_passed = False
                
                # Check if all metrics are computed
                required_metrics = ["Range", "Mean", "Mode", "M_a", "M_b"]
                for metric in required_metrics:
                    if metric in result.stdout:
                        print(f"✅ Metric '{metric}' computed")
                    else:
                        print(f"❌ Metric '{metric}' missing")
                        all_checks_passed = False
            else:
                print(f"❌ Script failed with exit code {result.returncode}")
                print(f"   Error: {result.stderr}")
                all_checks_passed = False
        except subprocess.TimeoutExpired:
            print("❌ Script timed out (>30 seconds)")
            all_checks_passed = False
        except Exception as e:
            print(f"❌ Error running script: {e}")
            all_checks_passed = False
    print()
    
    # Check 4: All 5 figures generated
    print("4. Checking generated figures...")
    figures = [
        "figures/fig1_hist_shares.png",
        "figures/fig2_boxplot_shares.png",
        "figures/fig3_correlation_heatmap.png",
        "figures/fig4_scatter_plot.png",
        "figures/fig5_bar_chart.png"
    ]
    
    for fig in figures:
        if check_file_exists(fig, f"Figure: {fig}"):
            # Check file size (should be > 1KB)
            size = os.path.getsize(fig)
            if size > 1024:
                print(f"   Size: {size / 1024:.1f} KB")
            else:
                print(f"   ⚠️  Warning: File size is very small ({size} bytes)")
        else:
            all_checks_passed = False
    print()
    
    # Summary
    print("=" * 70)
    if all_checks_passed:
        print("✅ ALL CHECKS PASSED - Project is ready for submission!")
    else:
        print("❌ SOME CHECKS FAILED - Please review the errors above")
    print("=" * 70)
    
    return all_checks_passed


if __name__ == "__main__":
    try:
        success = validate_project()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Validation error: {e}")
        sys.exit(1)
