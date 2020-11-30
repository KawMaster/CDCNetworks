# Libraries

#%% Load Libraries and set up script
import pandas as pd
import scipy.io as sio
import glob
import os

os.chdir('..')

#%% Reading in `.mat` data files and converting them.
mat = []
filenames = []
for filename in glob.glob('./data/raw/*.mat'):
    filenames.append(filename)
    with open(os.path.join('.', filename), 'r') as f:
        m = sio.loadmat(filename, squeeze_me = True)
        mat.append(m['U'])

# %%
for m,f in zip(mat, filenames):
    fn = f.split('\\')[1].split('.mat')[0]
    df = pd.DataFrame(m)
    dir_name = './data/processed/' + fn + '.csv'
    df.to_csv(dir_name, index = False)
