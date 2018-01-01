import numpy as np
import os

###global gravity constant###
G = 6.674 #  * 10**-11


########FUNCTIONS##########

def distances(obj_index, positions):
    """computes geometric distance from selected object to all other objects inc. self (which == 0)"""
    # print(np.linalg.norm(positions[obj_index] - positions, axis = 1))
    return np.linalg.norm(positions[obj_index] - positions, axis = 1)

def forces(obj_index, positions, masses):  #######improve efficiency by slicing out self? instead of dropping 0?
    """returns force in Newtons for each object to each other object, no direction information"""
    dists = distances(obj_index, positions)     #calcs distances for given object, by index
    force_list = G*masses[obj_index]*masses / (dists**2) #Vector of G of mass times masses, including self, unit divided by dist ^2

    # force_list = -G / (dists**3)

    #force_list = np.multiply(force_num,dists**-2)
    force_list[obj_index]=0  # removes division by 0 caused by 0-dist to self in above line
    return force_list

def unit_vectors(obj_index, positions):
    """returns unit vector for obj to each other object"""
    un_normed = positions - positions[obj_index]
    norming = np.linalg.norm(un_normed, axis=1) ** -1  #abs values of array for suming
    normed = norming * np.transpose(un_normed)
    normed = normed.transpose()
    return normed   ######dbl check math on normalization

def net_force_vector(obj_index, positions, masses):
    """computes x,y,z force vector, in Newtons, for given object index"""
    unit_vector = unit_vectors(obj_index, positions)  #has NaN for obj_index
    force_mags = forces(obj_index, positions, masses)
    force_mags = force_mags * unit_vector.transpose()
    force_mags = force_mags.transpose()
    force_mags = force_mags[(force_mags == force_mags).all(1)]
    force_mags = np.sum(force_mags, axis = 0)
    return force_mags

    # relative position array (pos array - pos[index]
    # solve geometry, x/y y/z z/x (etc.) * force to return obj by 3(dim) vectorized force

def delta_VP(velocities, positions, masses, time_slice):
    """returns two numpy arrays in list Velocities
    and Positions updated according to time_slice in seconds"""
    for index, _ in enumerate(positions):  #better way to access index?
        force_v = net_force_vector(index,positions,masses)
        velocities[index] = velocities[index] + (force_v * time_slice / masses[index])   ##delta impulse /mass = delta velocity + orig = new velocity, ammended to input
    positions += velocities * time_slice ### pos * velocity * time = delta position
    return [velocities,positions]

def discrete_simulation(velocities, positions, masses, time_slice = .0001, time_max = 50, verbose = False):
    """returns new velocity, positions after iterating at a delta_time of time_slice
    until time_max is reached.  Interval states are discarded"""
    steps = int(time_max/time_slice)
    for _ in range(steps):
        [velocities, positions] = delta_VP(velocities,positions,masses,time_slice)
        if verbose:
            # print(str(positions) + 'positions')
            print(str(velocities)+ 'vel')
    return [velocities, positions]

def continuous(velocities, positions, masses, interval, steps, serial, time_slice = 0.1):
    """Calls discrete simulation n = steps of time interval length with time_slice resolution.
    Stores and outputs a csv V/P arrays hstackd indexed by step #"""
    for step in range(steps):
        save_state([masses[:,None], velocities, positions],step*interval, serial)

        [velocities, positions] = discrete_simulation(velocities, positions, masses, time_slice, interval)


def save_state(M_V_P, time, serial):
    combined = np.hstack(M_V_P)
    folder_path = str('data\\system_'+str(serial))

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    file_name = folder_path + str('\\time_' + str(time) +'.csv')
    np.savetxt(file_name, combined, delimiter=',', header = str(time))



    #  print(force_list)
### Testing ###
# pos_array[2,1]=1.876
# pos_array[0,2]=5.867546546
# pos_array[4,0]=4.3


# a = distances(0,pos_array)
# print(a)


#####Need data storage, need plotting
##### need to simulate known system, eg earth around sun, moon around earth.
##### want to try pypy