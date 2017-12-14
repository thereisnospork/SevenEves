import numpy as np
from timeit import default_timer as timer

####numerical solver 3 body ###
### F = G m1*m2/d^2
start = timer()
n_bodies = 30  #bodies
dim = 3  # (x,y) GLOBAL
G = 6.674 * 10**-11

# pos_array= np.zeros([n_bodies,dim], dtype=np.float64) # position relative in meters, x,y,z to arbitrary origin for each of n bodies
velocities = np.zeros([n_bodies,dim], dtype=np.float32)  # m/s x,y for each of n bodies
# mass_array = np.ones([n_bodies], dtype=np.float32)  # masses for each of n bodies, in kg
#force_array = np.zeros([n_bodies,dim], dtype=np.float64)  # net forces in x,y for each of n bodies  ##extraneous

###initialize with psuedo random values ##need to research approx astronomical distributions, suns, planets, etc.
pos_array   =   np.random.normal(100,10,[n_bodies,dim])             #, dtype=np.float64) # position relative in meters, x,y,z to arbitrary origin for each of n bodies
# velocities  =   np.random.normal(100,10,[n_bodies,dim])             #, dtype=np.float64)  # m/s x,y for each of n bodies
mass_array  =   100* np.random.normal(100,10,[n_bodies])        #, dtype=np.float32)  # masses for each of n bodies, in kg


########FUNCTIONS##########

def distances(obj_index, positions):
    """computes geometric distance from selected object to all other objects inc. self (which == 0)"""
    out = np.linalg.norm(positions[obj_index] - positions, axis = 1)
    return out

def forces(obj_index, positions, masses):  #######DEBUG ME error line 36
    """returns force in Newtons for each object to each other object, no direction information"""
    dists = distances(obj_index, positions)     #calcs distances for given object, by index
    force_list = G*masses[obj_index]*masses / (dists**2) #Vector of G of mass times masses, including self, unit divided by dist ^2
    #force_list = np.multiply(force_num,dists**-2)
    force_list[obj_index]=0  # removes division by 0 caused by 0-dist to self in above line
    return force_list

def unit_vectors(obj_index, positions):
    """returns unit vector for obj to each other object"""
    un_normed = positions - positions[obj_index]
    norming = np.sum(np.absolute(un_normed), axis=1) ** -1  #abs values of array for suming
    normed = norming * np.transpose(un_normed)
    normed = normed.transpose()
    return normed

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
        velocities[index] += force_v * time_slice / mass_array[index]        ##delta impulse /mass = delta velocity + orig = new velocity, ammended to input
    positions += positions * velocities * time_slice ### pos * velocity * time = delta positionprobably needs a math/python debugging
    return [velocities,positions]

def discrete_simulation(velocities, positions, masses, time_slice = 0.1, time_max = 100):
    steps = int(time_max//time_slice)
    for _ in range(steps):
        [velocities, positions] = delta_VP(velocities,positions,masses,time_slice)
    return [velocities, positions]

def continuous(velocities, positions, masses, interval, steps, time_slice = 0.1):
    out = dict()
    for _ in range(steps):
        [velocities, positions] = discrete_simulation(velocities, positions, masses, time_slice, interval)
        out[(steps*interval)] = velocities,positions
    return out

    #  print(force_list)
### Testing ###
# pos_array[2,1]=1.876
# pos_array[0,2]=5.867546546
# pos_array[4,0]=4.3


# a = distances(0,pos_array)
# print(a)


c = continuous(velocities, pos_array, mass_array,10,1000,time_slice=0.1)

end = timer()

print(end - start)

print(c)

#####Need data storage, need plotting
##### need to simulate known system, eg earth around sun, moon around earth.
##### want to try pypy