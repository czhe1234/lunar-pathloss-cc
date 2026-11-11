import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cc_core import gpu_guard
print(gpu_guard(),flush=True)
