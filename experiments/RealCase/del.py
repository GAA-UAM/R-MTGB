# %%
import sys
import pandas as pd
from pathlib import Path



current_file_path = Path(__file__).resolve()
script_dir = current_file_path.parent
project_root = script_dir.parents[1]

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
    
from DataUtils.read_data import *


x_train, y_train, x_test, y_test = ReadData(dataset="avila", random_state=int(1))

# %%
x_train.shape[0] + x_test.shape[0]
# len(list(set(x_train[:, -1])))
x_train.shape