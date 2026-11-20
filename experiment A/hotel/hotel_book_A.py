"""Hotel Experiment A entry point. See experiment A/README.md."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_a import main

if __name__ == "__main__":
    main("hotel")
