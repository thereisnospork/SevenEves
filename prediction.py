import tensorflow as tf
import graphing as graph
import os
import numpy as np
from timeit import default_timer as timer

###First import data, graph has some functions
start = timer()



directory = 'sub_data\\'
sys_names = os.listdir(directory)
directories = (directory + x + '\\' for x in sys_names)

###Reading from disc and slicing into start conditions and output VPs
outs = [graph.files_to_MVP(directory) for directory in directories]
ins = [graph.MVP_start(MVP) for MVP in outs]

for i, each in enumerate(ins):
    ins[i] = np.dstack([ins[i]]*len(outs[0][0,0,:])) # extends initial conditions along 3rd axis equal to number of time points
    ins[i]=ins[i].flatten()

for i, each in enumerate(outs):
    outs[i]=outs[i].flatten()




# print(len(outs[0]))
# print(len(ins[0]))
#
#

num_bodies = 5
num_sys_in = len(ins)
num_points = len(ins[0])
# num_cols = 7  ###len(a[0][0,:,0]

# ####assemble feed dict up here!!!!!!!!


sess = tf.InteractiveSession()

x = tf.placeholder


x = tf.placeholder(tf.float32, shape = [None, num_points]) #[None,num_bodies,num_cols,num_points], name = 'x')  #[None, dimensions of tensor derived from starting conditions
y_ = tf.placeholder(tf.float32, shape =[None, num_points]) #num_bodies,num_cols,num_points], name = 'y_')


layer1=tf.layers.dense(x, num_points, tf.nn.relu)
y = tf.layers.dense(layer1, num_points, tf.nn.relu)

print(y)
print(layer1)



with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits = y, labels = y_)
    )
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = 0#tf.train.RMSPropOptimizer(0.25, momentum = 0.5).minimize(cross_entropy)


if True: #do_training == 1:
    sess.run(tf.global_variables_initializer())

    for i in range(11): #10001
        if i % 1 == 0: #batch size
            # print(ins)
            # print(ins[0])
            # print(outs[0])
            print(i)
            train_error = cross_entropy.eval(feed_dict={x: ins, y_: outs})
            print('step %d, training error %g') % (i, train_error)
            # if train_error < 0.0005:
            #     break
#
#         # if write_for_tensorboard == 1 and i % 5 == 0:
#         #     s = sess.run(merged_summary, feed_dict={x:ins[i], y_: outs[i]})
#         #     writer.add_summary(s, i)
#
#         sess.run(train_step, feed_dict={x: ins, y_: outs})
#
#         print('Test error using test data %g except not really right now' % (cross_entropy.eval(feed_dict={x:outs, y_:outs})))
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
end = timer()
print(end-start)