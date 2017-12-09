import numpy as np
from timeit import default_timer as timer

####numerical solver 3 body ###
### F = G m1*m2/d^2
start = timer()
n_bodies = 6  #bodies
dim = 2  # (x,y) GLOBAL
G = 6.674 * 10**-11

pos_array= np.zeros([n_bodies,dim]) # position relative in meters, x,y to arbitrary origin for each of n bodies
velocities = np.zeros([n_bodies,dim])  # m/s x,y for each of n bodies
mass_array = np.ones([n_bodies])  # masses for each of n bodies, in kg
force_array = np.zeros([n_bodies,dim])  # net forces in x,y for each of n bodies

########FUNCTIONS##########

def distances(obj_index, positions):
    """computes geometric distance from selected object to all other objects"""
    n = len(positions)
    out = np.zeros([n])
    for index, each in enumerate(positions):
        out[index] = np.linalg.norm(positions[obj_index]-each)
        #  print(out)
    return(out)

def forces(obj_index, positions, masses):
    """returns force in Newtons for each object to each other object, no direction information"""
    dists = distances(obj_index, positions)     #calcs distances for given object, by index
    force_list = G*masses[obj_index]*masses / dists**2 #Vector of G of mass times masses, including self, unit divided by dist ^2
    #force_list = np.multiply(force_num,dists**-2)
    force_list[obj_index]=0  # removes division by 0 caused by 0-dist to self in above line
    return force_list

def force_vectors(obj_index, positions, force_mags):
    """returns net x/y/z components of force for an object as a 3-unit vector"""
    # relative position array (pos array - pos[index]
    # solve geometry, x/y y/z z/x (etc.) * force to return obj by 3(dim) vectorized force 



    #  print(force_list)
### Testing ###
pos_array[2,1]=1.876
pos_array[0,1]=5.867546546
pos_array[0,0]=4.3


for i in range(1):
    force_output = forces(i, pos_array, mass_array)
    print(force_output)


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