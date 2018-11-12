import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

EPS = 1e-12
NUM_EPOCHS = 5000
BATCH_SIZE = 32
LEARNING_RATE = 0.001

def generate_data(n=1000, scale=10):
    X = scale*(np.random.random_sample(n) - 0.5)
    Y = X**2 + 5

    return np.array([X, Y]).T

def random_mini_batches(data, batch_size):
    mini_batches = []
    m = data.shape[0]
    np.random.shuffle(data)

    # Partition into mini-batches
    num_complete_batches = math.floor(m / batch_size)
    for i in range(num_complete_batches):
        batch = data[i * batch_size : (i + 1) * batch_size]
        mini_batches.append(batch)

    # Handling the case that the last mini-batch < batch_size
    if m % batch_size != 0:
        batch = data[num_complete_batches * batch_size : m]
        mini_batches.append(batch)

    return mini_batches

def generate_Z_batch(batch_size):
    return np.random.normal(size=(batch_size, 2))

def plot_data(data):
    plt.plot(data[:, 0], data[:, 1], "o")
    plt.savefig("output/data.png", bbox_inches="tight")
    plt.clf()

def plot_generated_data(data, gen_func, epoch):
    Z_batch = generate_Z_batch(data.shape[0])
    gen_data = gen_func(Z_batch)
    plt.plot(data[:, 0], data[:, 1], "o")
    plt.plot(gen_data[:, 0], gen_data[:, 1], "o")
    plt.title("Epoch " + str(epoch))
    plt.savefig("output/gen_data_" + str(epoch) + ".png", bbox_inches="tight")
    plt.clf()

def generator(Z, hidden_sizes=[10, 10]):
    with tf.variable_scope("GAN/Generator"):
        h1 = tf.layers.dense(Z, hidden_sizes[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hidden_sizes[1], activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2, 2)

    return out

def discriminator(X, hidden_sizes=[10, 10], reuse=False):
    with tf.variable_scope("GAN/Discriminator", reuse=reuse):
        h1 = tf.layers.dense(X, hidden_sizes[0], activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1, hidden_sizes[1], activation=tf.nn.leaky_relu)
        out = tf.layers.dense(h2, 1, activation=tf.nn.sigmoid)

    return out

def create_model(X, Z):
    gen_out = generator(Z)
    disc_out_real = discriminator(X)
    disc_out_fake = discriminator(gen_out, reuse=True)

    return gen_out, disc_out_real, disc_out_fake

# Generate training data
data = generate_data()
plot_data(data)
mini_batches = random_mini_batches(data, BATCH_SIZE)

# Create model
X = tf.placeholder(tf.float32, [None, 2])
Z = tf.placeholder(tf.float32, [None, 2])
gen_out, disc_out_real, disc_out_fake = create_model(X, Z)

# Create loss functions
disc_loss = tf.reduce_mean(-(tf.log(disc_out_real + EPS) + tf.log(1 - disc_out_fake + EPS)))
gen_loss = tf.reduce_mean(-tf.log(disc_out_fake + EPS))

# Get variables
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="GAN/Discriminator")

# Create training steps
gen_train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(gen_loss, var_list=gen_vars)
disc_train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(disc_loss, var_list=disc_vars)

# Start session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Training loop
    for epoch in range(NUM_EPOCHS):
        for X_batch in mini_batches:
            Z_batch = generate_Z_batch(X_batch.shape[0])
            _, disc_loss_train = sess.run([disc_train_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
            _, gen_loss_train = sess.run([gen_train_step, gen_loss], feed_dict={Z: Z_batch})
        if (epoch % 250) == 0:
            print("Epoch " + str(epoch) + " - plotting generated data")
            plot_generated_data(data, lambda Z_batch: sess.run(gen_out, feed_dict={Z: Z_batch}), epoch)

    print("Training finished - plotting generated data")
    plot_generated_data(data, lambda Z_batch: sess.run(gen_out, feed_dict={Z: Z_batch}), NUM_EPOCHS - 1)
