import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)
# Discriminator
X = tf.placeholder(tf.float32, shape=[None, 784], name = 'x')

#Generator
Z = tf.placeholder(tf.float32, shape=[None, 100], name = 'Z')

def generator(z):
    with tf.variable_scope("generator", reuse=False):
        gdl1 = tf.layers.dense(inputs=z,
                               units=128,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               bias_initializer=tf.zeros_initializer(),
                               activation=tf.nn.relu)
        gdl2 = tf.layers.dense(inputs=gdl1,
                               units=784,
                               kernel_initializer=tf.contrib.layers.xavier_initializer(),
                               bias_initializer=tf.zeros_initializer(),
                               activation=tf.sigmoid)
    return gdl2

def discrimator(x, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        dl1 = tf.layers.dense(inputs=x,
                               units=128,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              bias_initializer=tf.zeros_initializer(),
                               activation=tf.nn.relu)
        dl2 = tf.layers.dense(inputs=dl1,
                               units=1,
                              kernel_initializer=tf.contrib.layers.xavier_initializer(),
                              bias_initializer=tf.zeros_initializer(),
                               activation=tf.sigmoid)
    return dl2

G_sample = generator(Z)

D_real = discrimator(X)
D_fake = discrimator(G_sample, reuse=True)

D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
G_loss = -tf.reduce_mean(tf.log(D_fake))

g_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
d_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")

print(g_vars)
print(d_vars)

D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list = d_vars)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list = g_vars)

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m,n])

mb_size = 128
Z_dim = 100
mnist = input_data.read_data_sets('mnist', one_hot=True)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

if not os.path.exists('out/'):
    os.makedirs('out/')

sess = tf.Session()
sess.run(tf.global_variables_initializer())

i = 0
for it in range(1000000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, _ = mnist.train.next_batch(mb_size)
    _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
