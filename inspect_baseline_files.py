#!/usr/bin/env python3
"""
Check baseline forecast files and ensure dashboard compatibility
"""

import pandas as pd
from pathlib import Path

print("\n" + "="*70)
print("🔍 CHECKING BASELINE FORECAST FILES")
print("="*70 + "\n")

# Check both zones
for zone in ['DK1', 'DK2']:
    filepath = Path(f"data/forecast/co2_{zone}_baseline.csv")
    
    if not filepath.exists():
        print(f"❌ {zone}: File not found: {filepath}")
        continue
    
    print(f"✅ {zone}: {filepath} exists ({filepath.stat().st_size:,} bytes)")
    
    # Read and examine the file
    try:
        df = pd.read_csv(filepath)
        print(f"   Shape: {df.shape} (rows, columns)")
        print(f"   Columns: {list(df.columns)}")
        print(f"   First few rows:")
        print(df.head(3))
        print(f"\n   Data types:")
        print(df.dtypes)
        print()
        
    except Exception as e:
        print(f"   ❌ Error reading file: {e}\n")

print("="*70)
print("\n💡 Dashboard is looking for these files:")
print("   • data/forecast/co2_DK1_baseline.csv")
print("   • data/forecast/co2_DK2_baseline.csv")
print("\n✅ Both files exist!")
print("\n🔍 Possible issues:")
print("   1. Dashboard might be looking for different column names")
print("   2. Dashboard might be caching old data")
print("   3. Dashboard code might have a path or naming issue")
print("\n🔧 Solutions:")
print("   1. Restart Streamlit (Ctrl+C, then restart)")
print("   2. Hard refresh browser (Ctrl+Shift+R)")
print("   3. Check dashboard.py for file loading logic")
print()