import numpy as np
import main
from timeit import default_timer as timer

###Init test arrays (0's, 1's, etc.)

def init_gaus(interval, years, iterations,serial_num = 0, time_slice=0.1, n_bodies = 3, dim = 3):

###initialize with gaus values ##need to research approx astronomical distributions, suns, planets, etc.
    # n_bodies = 3  # bodies
    # dim = 3  # (x,y,z)
 #   G = 6.674 * 10 ** -11
 #    s_per_hour = 3600
 #    s_per_week = 604800
    s_per_mon = 2629800
    s_per_year = 12 * s_per_mon

    steps = int(years * s_per_year // interval)

    for i in range(iterations):
        start = timer()
        # pos_array= np.zeros([n_bodies,dim], dtype=np.float64) # position relative in meters, x,y,z to arbitrary origin for each of n bodies
        # velocities = np.zeros([n_bodies,dim], dtype=np.float32)  # m/s x,y for each of n bodies
        # mass_array = np.ones([n_bodies], dtype=np.float32)  # masses for each of n bodies, in kg
        # # force_array = np.zeros([n_bodies,dim], dtype=np.float64)  # net forces in x,y for each of n bodies  ##extraneous

        pos_array   =   100 * np.random.normal(0,10,[n_bodies,dim])             #, dtype=np.float64) # position relative in meters, x,y,z to arbitrary origin for each of n bodies
        velocities  =   0 * np.random.normal(0,25,[n_bodies,dim])             #, dtype=np.float64)  # m/s x,y for each of n bodies
        mass_array  =   10**2 * np.random.normal(10,1,[n_bodies])        #, dtype=np.float32)  # masses for each of n bodies, in kg

        mass_array = mass_array *2000


        print('initial conditions')
        print(str(pos_array)+' pos')
        print(str(velocities)+' velocities')


        pos_array.astype(np.float64)
        velocities.astype(np.float64)
        mass_array.astype(np.float64)

        # mass_array = 10**10 * mass_array
        # velocities = 5 * velocities
        # pos_array = 10**5 * pos_array


        main.continuous_2(velocities,pos_array,mass_array,interval,steps,serial_num, time_slice)
        serial_num += 1

        end = timer()
        print(str(end-start) + 's elapsed this loop')


init_gaus(3600, .1, 99, time_slice=1, serial_num = 30005, dim = 3, n_bodies=5)



