import matplotlib as mpl
import bokeh as bh
import numpy as np
import os


def files_to_times(files):
    times = list()
    for title in files:
        times.append(int(title.split('.')[0].split('_')[1]))
    return times

def files_to_MVP(files):
    """pulls all the files in the directory into one large cubic numpy array"""
    body_count = num_bodies(files)
    master = np.zeros((body_count,7,len(files))) ##ROW | COLS | TIME
    for index, file in enumerate(files):
        master[:,:,index] = np.genfromtxt('data\\'+system +'\\'+ file, delimiter=',')
    return master

def num_bodies(files):
    a = np.genfromtxt('data\\'+system +'\\'+ files[0])
    return(len(a))

def masses(MVP):
    return MVP[:,1,:]

def velocities(MVP):
    return MVP[:,1:4,:]

def positions(MVP):
    return MVP[:,4:8,:]





system = 'system_0'
data_files = os.listdir('data\\'+ system+'\\')

foo = files_to_times(data_files)
bar = num_bodies(data_files)
asdf = files_to_MVP(data_files)
print(asdf[:,4:8,0])