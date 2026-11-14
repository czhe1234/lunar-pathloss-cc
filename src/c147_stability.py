"""C147 bounded stability checks; reuse frozen inference worker unchanged."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import c147_paths
mode=sys.argv.pop(1)
assert mode in ("sampling262144","totaldepth12")
c147_paths.A=ROOT/"audit/c147_sionna_reflections"/mode
c147_paths.main()
