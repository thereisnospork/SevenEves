import tensorflow as tf
# import graphing as graph
import os
import numpy as np
from timeit import default_timer as timer

###First import data, graph has some functions
start = timer()

def ins_outs(directory):
    """reads .npy files from directory into (ins,outs) tuple of two lists  of 1-d np arrays
    each consisting of one full system that has been flattened.  The ins are system t0's
    expanded to match the dimensionality of the outs
    Also normalizes along axis 1 (masses, Vx, Vy, etc. for each system and returns list of
    norming arrays for data reconstitution"""

    files = os.listdir(directory)
    outs = [np.load(directory + file_) for file_ in files]
    outs = [out[:,:,0:50] for out in outs] #limit to first 50 time stamps for expedited computation/memory req.
    outs = [out / np.linalg.norm(out, axis = 0, ord = 1) for out in outs]

    norms=list()

    for index, out in enumerate(outs):
        norm = np.linalg.norm(out, axis = 0)
        norms.append(norm)
        out[np.isnan(out)] = 0  #scrub Nans from divide by 0

    ins = [out[:, :, 0] for out in outs]

    for i, each in enumerate(ins):
        ins[i] = np.dstack([ins[i]] * len(outs[0][0, 0, :]))  # extends initial conditions along 3rd axis equal to number of time points
        ins[i] = ins[i].flatten()

    for i, each in enumerate(outs):
        outs[i] = outs[i].flatten()

    ins = np.array(ins)
    outs = np.array(outs)
    return ins, outs, norms

directory = 'data_out\\data2\\'
ins, outs, norms = ins_outs(directory)

# ins_test = np.ones([1,30660])#test value for seperate test/training error
# outs_test = np.ones([1,30660])


num_bodies = 5
num_sys = len(outs)
num_in = len(ins[0])
num_points = len(outs[0])

# num_cols = 7  ###len(a[0][0,:,0]

print(num_in)
print(num_points)

############        MODEL        ###############

sess = tf.InteractiveSession()

# x = tf.placeholder


x = tf.placeholder(tf.float32, shape = [None, num_points]) #[None,num_bodies,num_cols,num_points], name = 'x')  #[None, dimensions of tensor derived from starting conditions
y_ = tf.placeholder(tf.float32, shape =[None, num_points]) #num_bodies,num_cols,num_points], name = 'y_')



layer1 = tf.layers.dense(x, num_points, tf.nn.relu)
layer2 = tf.layers.dense(layer1, num_points, tf.nn.relu)
layer3 = tf.layers.dense(layer2, num_points, tf.nn.relu)
layer4 = tf.layers.dense(layer3, num_points, tf.nn.relu)
layer5 = tf.layers.dense(layer4, num_points, tf.nn.relu)
layer6 = tf.layers.dense(layer5, num_points, tf.nn.relu)
layer7 = tf.layers.dense(layer5, num_points, tf.nn.relu)
layer8 = tf.layers.dense(layer6, num_points, tf.nn.relu)
layer9 = tf.layers.dense(layer7, num_points, tf.nn.relu)
layer10 = tf.layers.dense(layer8, num_points, tf.nn.relu)
layer11 = tf.layers.dense(layer9, num_points, tf.nn.relu)
layer12 = tf.layers.dense(layer10, num_points, tf.nn.relu)
layer13 = tf.layers.dense(layer11, num_points, tf.nn.relu)
layer14 = tf.layers.dense(layer12, num_points, tf.nn.relu)
layer15 = tf.layers.dense(layer13, num_points, tf.nn.relu)
layer16 = tf.layers.dense(layer14, num_points, tf.nn.relu)
layer17 = tf.layers.dense(layer15, num_points, tf.nn.relu)
y       = tf.layers.dense(layer16, num_points, tf.nn.relu)
###replace masses in y with initial values manually here, by shape/reshape/replace/deshape

# y = tf.reshape(y,[5,7,50]) #need to generalize for other dimensions!
# x = tf.reshape(x,[5,7,50])
#
# weights = x[0,:,:] #similar to graph.masses()
#
# y = tf.stack([weights,y[1,:,:]], axis=1)
#
# print(tf.shape(weights))
# print(tf.shape(y))
# print(layer1)
# print(y)



with tf.name_scope('cross_entropy'):
    # cross_entropy = tf.nn.l2_loss(y_-y)
    # cross_entropy = tf.losses.huber_loss(labels = y_, predictions = y)
    cross_entropy = tf.losses.absolute_difference(labels = y_, predictions = y)
    # cross_entropy = tf.reduce_sum(abs(y_-y)/num_sys)

    # sum_ = tf.reduce_sum(y_)/num_sys


    # cross_entropy = tf.reduce_mean(
    #     abs(tf.nn.sigmoid_cross_entropy_with_logits(logits = y, labels = y_))
    # )
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    # train_step = tf.train.GradientDescentOptimizer(.5).minimize(cross_entropy)
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy) #uses more memory> caching bad with full t
    # train_step = tf.train.AdadeltaOptimizer().minimize(cross_entropy)

    # __init__(
    #     learning_rate=0.001,
    #     beta1=0.9,
    #     beta2=0.999,
    #     epsilon=1e-08,
    #     use_locking=False,
    #     name='Adam'


if True: #do_training == 1:


    sess.run(tf.global_variables_initializer())
    train_i = list(range(len(ins))) #first batch on whole system.  need to have a value here before called

    for i in range(200001): #10001
        loop_start = timer()
        sess.run(train_step, feed_dict={x: ins[train_i], y_: outs[train_i]})

        if i % 500 == 0: #batch size
            test_i = list()
            train_i = list()
            
            for index, _ in enumerate(ins):
                p = np.random.random()
                if p > 0.6: #split train : test
                    test_i.append(index)
                else:
                    train_i.append(index)

            train_error = cross_entropy.eval(feed_dict={x: ins[train_i], y_: outs[train_i]})
            test_error = cross_entropy.eval(feed_dict={x: ins[test_i], y_: outs[test_i]})
            # print(str(sum_.eval(feed_dict ={y_: outs}))+'avg sum of single system')

            # print('test error:{0}'.format(test_error))
            # if train_error < 0.0005:
            #     break

            loop_end = timer()
            delta_t = loop_end - loop_start
            print('step {0}, training error {1}, test error {2} in {3}-seconds'.format(i, train_error, test_error, delta_t))


#


    asdf = y.eval(feed_dict={x: np.reshape(ins[0],[1,1750])})
    asdf = np.reshape(asdf,[5,7,50])
    foo = np.reshape(ins[0],[5,7,50])
    np.savetxt('out\\out_testing.csv',asdf[:,:,0])
    np.savetxt('out\\out_testing5.csv',asdf[:,:,5])
    np.savetxt('out\\in_testing.csv',foo[:,:,0])
    np.savetxt('out\\in_testing5.csv', foo[:, :, 5])
    np.save('out\\full_diff.npy', (foo-asdf))
    print('difference' + np.sum(abs(foo)-abs(asdf)))
    # print(type(asdf))
    # print(asdf.shape)
    # print(np.reshape(asdf,[5,7,876]))










#
        # print('Test error using test data %g' % (cross_entropy.eval(feed_dict={x:ins, y_:outs})))
#
#
#
#
# # print(ins[0].shape)
# # print(outs[0].shape)
# # print(type(ins[0][0,0]))
# #
#
# #
# # num_test = 0
# # num_train = 0
# #
# # ##############OPTIONS############
# #
# # write_for_tensorboard = 0
# # do_training = 1  # 1 = do the training, 0 = load from file and just run it
# # save_trained = 0  # 1 = save to file after training, 0 = don't save
# # tensorboard_file = '/out/testing_1'
# # save_file = '/out/testing_etc.ckpt'
# #
# #
# # sess = tf.InteractiveSession()
# #
# #
# # with tf.name_scope('cross_entropy'):
# #     cross_entropy = tf.reduce_mean(
# #         tf.nn.sigmoid_cross_entropy_with_logits(logits = y, labels = y_)
# #     )
# #     tf.summary.scalar('cross_entropy', cross_entropy)
# #
# # with tf.name_scope('train'):
# #     train_step = tf.train.RMSPropOptimizer(0.25, momentum = 0.5).minimize(cross_entropy)
# #
# # if write_for_tensorboard == 1:
# #     merged_summary = tf.summary.merge_all()
# #     print("Writing for TensorBoard to file %s" % (tensorboard_file))
# #     writer = tf.summary.FileWriter(tensorboard_file)
# #     writer.add_graph(sess.graph)
# #
# # if do_training == 1:
# #     sess.run(tf.global_variables_initializer())
# #
# #     for i in range(11): #10001
# #         if i % 1 == 0: #batch size
# #             # print(ins)
# #             # print(ins[0])
# #             # print(outs[0])
# #             train_error = cross_entropy.eval(feed_dict={x: outs[i], y_: outs[i]})
# #             print('step %d, training error %g') % (i, train_error)
# #             # if train_error < 0.0005:
# #             #     break
# #
# #         if write_for_tensorboard == 1 and i % 5 == 0:
# #             s = sess.run(merged_summary, feed_dict={x:ins[i], y_: outs[i]})
# #             writer.add_summary(s, i)
# #
# #         sess.run(train_step, feed_dict={x: ins, y_: outs})
# #
# #         print('Test error using test data %g except not really right now' % (cross_entropy.eval(feed_dict={x:outs, y_:outs})))
# #
# #     if save_trained == 1:
# #         print("Saving neural network to %s.*" % (save_file))
# #         saver = tf.train.Saver()
# #         saver.save(sess, save_file)
# #
# # # else:  # if we're not training then we must be loading from file
# # #
# # #     print("Loading neural network from %s" % (save_file))
# # #     saver = tf.train.Saver()
# # #     saver.restore(sess, save_file)
# # #     # Note: the restore both loads and initializes the variables
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # # print(len(ins))
# # # print(num_bodies)
# # # print(num_sys_in)
# # # print(num_points)
# #
# #
# # #
# # #        ]
# # # outs = ??? VP all time slices minus first
# # #
# # # # a = list(a)
# # # #
# # # #
# # # # print(a[0][0,0,:]) ##system | [col(mv or p)|row(particle) |time stamps
# # # # print(len(a[0][0,0,:]))
# # #
# # # # x = tf.placeholder(tf.float32, shape =[None,5,7,4383]) #None any dimension, batch size,
# # # # y_ = tf.placeholder(tf.float32, shape = [None,5,6,4383]) #no mass term in output
# # # #
# # # # W = tf.Variable(tf.zeros([784,10]))
# # # # b = tf.Variable(tf.zeros([10]))
# # #
# # #
# # #
# # # def weight_variable(shape):
# # #     initial = tf.truncated_normal(shape, stddev=0.1)
# # #     return tf.Variable(initial)
# # #
# # # def bias_variable(shape):
# # #     initial = tf.constant(0.1, shape = shape)
# # #     return tf.Variable(initial)
# # #
# # # def conv2d(x, W):
# # #     return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')
# # #
# # # # def max_pool_2x2(x): Nnot needed
# # #
# # # y = conv2d(a,1)
# # # print(y)
# # #
# # #

#
# def rand_dict_train_test(ins,outs, var_ind = x, var_dep = y_, p = 0.8):
#     """creates 2 feed_dicts for TF eval, one to train and one to test.  p chance to be in test. rest to train.
#     might add size limit later???
#     possible issues if empty dict returned"""
#     test_list = list()
#     in_out = list(zip(ins,outs)) #fu python
#     for index, each in enumerate(in_out):
#         rnd = np.random.random()
#         if rnd > p:
#                 train_val = in_out.pop(index)
#                 test_list.append(train_val)
#     in_out = tuple(zip(*in_out))
#     test_list = tuple(zip(*test_list))
#
#     train_dict = {var_ind:in_out[0], var_dep:in_out[1]}
#     test_dict = {var_ind:test_list[0], test_list:in_out[1]}
#
#     return train_dict, test_dict

end = timer()
print(end-start)