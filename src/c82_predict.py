"""Score the frozen C82 H/TX predictor through the common official-map generator."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.c82_inference import Predictor
from src.c76_predict import main
if __name__=='__main__':main(Predictor)
