from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os

#####ONLY WORKS FOR 3D systems as written!!!!!!!!!!




def files_to_times(files):
    times = list()
    for title in files:
        times.append(int(title.split('.')[0].split('_')[1]))
    return times

def files_to_MVP(directory):
    """pulls all the files in the directory into one large cubic numpy array"""
    files = os.listdir(directory)
    body_count = num_bodies(files, directory)
    master = np.zeros((body_count,7,len(files))) ##ROW | COLS | TIME
    for index, file in enumerate(files):
        master[:,:,index] = np.genfromtxt(directory + file, delimiter=',')
    return master

def num_bodies(files, directory):
    a = np.genfromtxt(directory + files[0])
    # print(len(a))
    return len(a)

def masses(MVP):
    return MVP[:,0,:]

def velocities(MVP):
    return MVP[:,1:4,:]

def positions(MVP):
    return MVP[:,4:8,:]

def VP_results(MVP):
    return MVP[:,1:8,1:-1] #excludes starting conditions

def MVP_start(MVP):
    return MVP[:,:,0]


def graph_pos(system,init=False, end = False):
    directory = 'data\\' + system + '\\'
    mvp = files_to_MVP(directory)
    # times = files_to_times(data_files)
    m = masses(mvp)
    # v = velocities(mvp)
    p = positions(mvp)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    if init:
        p_x, p_y, p_z = p[:,0,0], p[:,1,0], p[:,2,0] # graphs initial positions
        col = m[:,0].flatten()
    elif end:
        p_x, p_y, p_z = p[:,0,0], p[:,1,-1], p[:,2,-1] # graphs final positions
        col = m[:,0].flatten()
    else:
        p_x, p_y, p_z = p[:,0], p[:,1], p[:,2] # graphs trace of position over time (no t coloration/axis)
        col = m.flatten()


    ax.scatter(p_x, p_y, p_z, c=col, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(system+'_pos')
    #
    # print(np.ndim(p))
    # print(p[:,0,0])
    # print(p[:,1,0])

    #p[0,,] is [x,y,z]'s for one body
    plt.show()

def graph_vel(system,init=False, end = False):
    directory = 'data\\' + system + '\\'
    mvp = files_to_MVP(directory)
    # times = files_to_times(data_files)
    m = masses(mvp)
    v = velocities(mvp)
    # p = positions(mvp)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')

    if init:
        v_x, v_y, v_z = v[:,0,0], v[:,1,0], v[:,2,0] # graphs initial positions
        col = m[:,0].flatten()

    elif end:
        v_x, v_y, v_z = v[:,0,0], v[:,1,-1], v[:,2,-1] # graphs initial positions
        col = m[:,0].flatten()

    else:
        v_x, v_y, v_z = v[:,0], v[:,1], v[:,2] # graphs trace of position over time (no t coloration/axis)
        col = m.flatten()


    ax.scatter(v_x, v_y, v_z, c=col, marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(system+'_vel')
    # print(v)
    plt.show()

# system = 'system_0'
# directory = 'data\\'+system +'\\'
# data_files = os.listdir(directory)
#
# graph_pos('system_100')
# graph_vel('system_100')
#




# foo = files_to_times(data_files)
# bar = num_bodies(data_files,directory)
# asdf = files_to_MVP(data_files,directory)
# qwert = positions(asdf)
# print(qwert)