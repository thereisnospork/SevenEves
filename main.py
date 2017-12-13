import numpy as np
from timeit import default_timer as timer

####numerical solver 3 body ###
### F = G m1*m2/d^2
start = timer()
n_bodies = 6  #bodies
dim = 3  # (x,y) GLOBAL
G = 6.674 * 10**-11

pos_array= np.zeros([n_bodies,dim]) # position relative in meters, x,y,z to arbitrary origin for each of n bodies
velocities = np.zeros([n_bodies,dim])  # m/s x,y for each of n bodies
mass_array = np.ones([n_bodies])  # masses for each of n bodies, in kg
force_array = np.zeros([n_bodies,dim])  # net forces in x,y for each of n bodies

########FUNCTIONS##########

def distances(obj_index, positions):
    """computes geometric distance from selected object to all other objects"""
    n = len(positions)
    out = np.zeros([n])
    for index, each in enumerate(positions):
        out[index] = np.linalg.norm(positions[obj_index]-each)   #### de for loop with axis = 1 option
        #  print(out)
    return(out)

def forces(obj_index, positions, masses):
    """returns force in Newtons for each object to each other object, no direction information"""
    dists = distances(obj_index, positions)     #calcs distances for given object, by index
    force_list = G*masses[obj_index]*masses / dists**2 #Vector of G of mass times masses, including self, unit divided by dist ^2
    #force_list = np.multiply(force_num,dists**-2)
    force_list[obj_index]=0  # removes division by 0 caused by 0-dist to self in above line
    return force_list

def unit_vectors(obj_index, positions):
    """returns unit vector for obj to each other object"""
    un_normed = positions - positions[obj_index]           > 1
    norming = np.sum(np.absolute(un_normed), axis=1) ** -1  #abs values of array for summing
    normed = norming * np.transpose(un_normed)
    normed = normed.transpose()
    return normed

def net_force_vector(obj_index, positions, masses):
    """computes x,y,z force vector, in Newtons, for given object index"""
    unit_vector = unit_vectors(obj_index, positions)  #has NaN for obj_index
    print(unit_vector)
    force_mags = forces(obj_index, positions, masses)
    force_mags = force_mags * unit_vector.transpose()
    # print(force_mags)
    force_mags = force_mags.transpose()
    force_mags = force_mags[(force_mags == force_mags).all(1)]
    force_mags = np.sum(force_mags, axis = 1)

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




    #  print(force_list)
### Testing ###
pos_array[2,1]=1.876
pos_array[0,2]=5.867546546
pos_array[4,0]=4.3

b = net_force_vector(2,pos_array, mass_array)
# a = delta_VP(velocities, pos_array,mass_array, 1)

print(b)
#print(b[1])

# for i in range(4):
#     force_output = forces(i, pos_array, mass_array)
#     print(force_output)

#print(unit_vectors(0,pos_array))

# def forces_all(obj_array,)



# foo = distances(1,pos_array)
# print(foo)

# print(forces(1, pos_array, mass_array))


# print(len(pos_array))

#
# asdf = forces(2,positions,mass_array)
#
#
# print(asdf[0:4])


end = timer()

print(end - start)

##asdf = distances(2,positions)


###distance = np.linalg.norm(pos1 - pos2)

# ########def forces(obj_index, positions, masses):
#     dists = distances(obj_index,positions)
#     force_list = G*1*2 / dists**2
#
#     print(force_list)



# def forces(obj_index, positions, masses):
#     """returns force in Newtons for each object to each other object, no direction information"""
#     dists = distances(obj_index, positions)     #calcs distances for given object, by index
#     force_list = G*masses[obj_index]*masses / dists**2 #Vector of G of mass times masses, including self, unit divided by dist ^2
#     force_list[obj_index]=0  # removes division by 0 caused by 0-dist to self in above line
#     print(masses/dists)
#     return force_list