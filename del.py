#%%


from scipy.io import loadmat
import numpy as np
# Load the .mat file
data = loadmat(r"D:\Post_PhD\Programming\Py\NoiseAwareBoost\Datasets\microarray\Processed_Microarray_Data.mat")

# See what's inside
print(data.keys())  # Shows all variable names in the .mat file
np.savetxt("microarray_target.csv", data['Y'], delimiter=",")
#%%
import os
os.getcwd()