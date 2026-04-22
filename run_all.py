"""run_all.py — Run the full backtesting pipeline."""
import subprocess, sys
from pathlib import Path

def run(cmd, desc):
    print(f"\n{'='*60}\n  {desc}\n{'='*60}")
    r = subprocess.run(cmd, shell=True, cwd=Path(__file__).parent)
    if r.returncode != 0:
        print(f"ERROR: failed with code {r.returncode}")
        sys.exit(r.returncode)

if __name__ == "__main__":
    print("\n📈 ALGO TRADING BACKTESTER — FULL PIPELINE")
    print("=" * 60)
    run("python data/fetch.py",        "STEP 1/3: Fetching market data")
    run("python backtest/engine.py",   "STEP 2/3: Running backtests for all strategies")
    run("python -m pytest tests/ -v",  "STEP 3/3: Running unit tests")
    print("\n" + "="*60)
    print("  ✅ ALL STEPS COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("  Dashboard:  streamlit run dashboard/app.py  → http://localhost:8501")
    print("  API:        uvicorn api.main:app --reload   → http://localhost:8000/docs")
    print("="*60)
