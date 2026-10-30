import sys
from pathlib import Path
sys.path.insert(0, "/mnt/c/Code/MARVEL_GNN/src")
import torch
torch.set_num_threads(2)
from marvel_gnn.gnn import train as T
seed = int(sys.argv[1])
_orig = torch.manual_seed
torch.manual_seed = lambda s: _orig(seed)   # only weight-init/dropout seed varies; data prep seeds unchanged
T.EPOCHS = 500
T.CO_DIR = Path("/mnt/c/Code/MARVEL_scraping/molecules/CO")
T.MODEL_PATH = Path(__file__).with_name(f"seed{seed}.pt")   # never touch the real checkpoint
T.main()
