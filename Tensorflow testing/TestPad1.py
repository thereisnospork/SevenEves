import tensorflow as tf

# node1 = tf.constant(3.0, dtype=tf.float32)
# node2 = tf.constant(4.0) #tf.float32 implied

#print(node1,node2)
#
sess = tf.Session()
# print(sess.run([node1,node2]))
# node3 = tf.add(node1,node2)
# print('node3:', node3)
# print('sess.run(node3):',sess.run(node3))


# a = tf.placeholder(tf.float32)
# b = tf.placeholder(tf.float32)
# adder_node = a + b # short cut for tf.add
# add_and_triple = adder_node * 3.
#
# print(sess.run(add_and_triple,{a: 3, b: 4.5}))

# print(sess.run(adder_node, {a: 3, b: 4.5}))
# print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))


W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3],dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
#

#
# print(sess.run(linear_model, {x: [1, 2, 3, 4]}))  #evaluates linear model over vales of x input
#

y =tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)

init = tf.global_variables_initializer()  #initialize variables with explicit command
sess.run(init)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))











# print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# fixW = tf.assign(W, [-1.])
# fixb = tf.assign(b, [1.])
# sess.run([fixW, fixb]) #2nd run cmd necessary to reassign variables
# print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

