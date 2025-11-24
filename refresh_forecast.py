#!/usr/bin/env python3
"""
Super Simple Live Forecast Updater
Just refreshes your existing ML forecasts with latest data
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

print("\n" + "="*70)
print("🔄 REFRESHING FORECASTS WITH LATEST DATA")
print("="*70 + "\n")

# Step 1: Update CO2 data
print("📡 Fetching latest CO₂ data from Energinet...")
try:
    result = subprocess.run(
        [sys.executable, "src/ingest/energinet_co2.py"],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    if result.returncode == 0:
        print("   ✅ CO₂ data updated successfully")
    else:
        print("   ⚠️  CO₂ update had warnings (using cached data)")
        if result.stderr:
            print(f"   Details: {result.stderr[:200]}")
except Exception as e:
    print(f"   ⚠️  Could not update CO₂ data: {e}")
    print("   Continuing with cached data...")

# Step 2: Run your ML forecast script
print("\n🤖 Generating ML forecasts...")

ml_script = Path("src/models/ml_forecast.py")

if not ml_script.exists():
    print("   ❌ ml_forecast.py not found!")
    print("   Expected location: src/models/ml_forecast.py")
    exit(1)

try:
    result = subprocess.run(
        [sys.executable, str(ml_script)],
        capture_output=True,
        text=True,
        timeout=120
    )
    
    if result.returncode == 0:
        print("   ✅ ML forecasts generated!")
        
        # Show output
        if result.stdout:
            for line in result.stdout.split('\n')[-10:]:  # Last 10 lines
                if line.strip():
                    print(f"   {line}")
    else:
        print("   ❌ ML forecast generation failed")
        print("\n   Error details:")
        if result.stderr:
            for line in result.stderr.split('\n')[:15]:  # First 15 lines
                print(f"   {line}")
        
        print("\n   💡 This might mean you need to train the model first.")
        print("   Your ml_forecast.py probably needs a trained model file.")
        exit(1)
        
except Exception as e:
    print(f"   ❌ Error running ml_forecast.py: {e}")
    exit(1)

# Step 3: Verify files were created
print("\n📋 Checking forecast files...")

forecast_found = False
for zone in ['DK1', 'DK2']:
    ml_file = Path(f"data/forecast/co2_{zone}_ml.csv")
    
    if ml_file.exists():
        size_kb = ml_file.stat().st_size / 1024
        mod_time = datetime.fromtimestamp(ml_file.stat().st_mtime)
        print(f"   ✅ {zone}: {ml_file.name} ({size_kb:.1f} KB, modified {mod_time.strftime('%H:%M:%S')})")
        forecast_found = True
        
        # Also create a "live" version
        live_file = Path(f"data/forecast/live_forecast_{zone}.csv")
        import shutil
        shutil.copy(ml_file, live_file)
        print(f"      → Copied to live_forecast_{zone}.csv")
    else:
        print(f"   ⚠️  {zone}: No forecast file found")

if not forecast_found:
    print("\n   ❌ No forecast files were created!")
    print("   Check if ml_forecast.py needs additional setup.")
    exit(1)

print("\n" + "="*70)
print("✅ FORECAST UPDATE COMPLETE")
print("="*70)
print("\n💡 Next steps:")
print("   1. Restart your dashboard if it's running")
print("   2. Go to ML Forecasts tab")
print("   3. You should see updated forecasts!")
print()