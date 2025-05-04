#%%
import os
import numpy as np
import pandas as pd

dataset = "computer"

for root, dir, files in os.walk(os.getcwd()):
    for file in files:
        if dataset in file:
            if file.endswith(".csv") and "y_test" in file:
                y_test_path = os.path.join(root, file)
            if file.endswith(".csv") and "y_train" in file:
                y_train_path = os.path.join(root, file)