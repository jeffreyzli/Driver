from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
import random

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train_dir', '/home/phillip/Projects/learntensorflow/mnist_beginner/', 'Directory for training model')
# flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')

# mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

# possible_entries = [[0, 1], [1, 0]]
training_x=[]
training_y=[]
num_samples = -1
with open('./nnrccar9404117983228546features', 'r') as fin:
    lines = fin.readlines()
    num_samples = len(lines) / 2
    for idx in range(0, len(lines)/2):
        outputs = lines[2 * idx].strip().split(" ")
        inputs = lines[2 * idx + 1].strip().split(" ")
        training_y.append([float(i) for i in outputs])
        training_x.append([float(i) for i in inputs])
# training_x = [[random.randint(0, 1) for i in range(3)] for j in range(1000)]
# training_y = [[0, 1] for j in range (1000)]
# testing_x = [[random.randint(0, 1) for i in range(3)] for j in range(100)]
# testing_y = [possible_entries[random.randint(0, 1)] for j in range(100)]
# print(training_y)
# print(testing_y)

# Creating model
x = tf.placeholder(tf.float32, [None, 25351])
W = tf.Variable(tf.zeros([25351, 5]))
b = tf.Variable(tf.zeros([5]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
# y = tf.Print(y, [y], message="This is y: ", summarize=10)

# Define placeholder for correct values
y_ = tf.placeholder(tf.float32, [None, 5])
# Define cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# Define train step
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Define initialization step, don't do it yet.
init = tf.initialize_all_variables()

# Launch model in session, run init
sess = tf.Session()
sess.run(init)

def getNextBatch(size):
    indexes = random.sample(range(0, num_samples), size)
    xs_test = [training_x[i] for i in indexes]
    ys_test = [training_y[i] for i in indexes]
    return xs_test, ys_test

# Run 1000 iterations of training step
for i in range(1000):
    if (i % 100 == 0):
        print(i)
    batch_xs, batch_ys = getNextBatch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print(sess.run(accuracy, feed_dict={x: testing_x, y_: testing_y}))

# create Saver object as normal in Python to save your variables
saver = tf.train.Saver()

# save at iteration "global step"
# saver.save(sess, FLAGS.train_dir, global_step=10)
saver_def = saver.as_saver_def()
print saver_def.filename_tensor_name
print saver_def.restore_op_name

# write out three files
saver.save(sess, 'trained_model.sd')
tf.train.write_graph(sess.graph_def, '.', 'trained_model.proto', as_text=False)
tf.train.write_graph(sess.graph_def, '.', 'trained_model.txt', as_text=True)
