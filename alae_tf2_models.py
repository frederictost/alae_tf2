import numpy as np
import tensorflow as tf

class F_encoder(tf.keras.layers.Layer):

    def __init__(self, conf_dict, tensorboard):
        super(F_encoder, self).__init__()
        self.latent_dim = conf_dict["LATENT_DIM"]
        self.z_dim      = conf_dict["Z_DIM"]
        self.image_dim  = conf_dict["IMAGE_DIM"]
        self.tensorboard = tensorboard
        self.step = 0
        self.input_layer = tf.keras.Input(self.z_dim)
        self.l1 = tf.keras.layers.Dense(1024, input_dim=self.image_dim, activation=tf.nn.relu)
        self.l2 = tf.keras.layers.Dense(512, activation=tf.nn.leaky_relu)
        self.output_f = tf.keras.layers.Dense(self.latent_dim)

        # self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        # self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        # self.batch_norm_3 = tf.keras.layers.BatchNormalization()

        self.the_output = self.the_call(self.input_layer)

    def call(self, x, training=True):
        return self.the_call(x, training=training)

    def the_call(self, x, training=True):
        x = self.l1(x)
        x = self.l2(x)
        f_output = self.output_f(x)

        self.step += 1
        if self.tensorboard is not None and self.step % 50 == 0:
            with self.tensorboard.as_default():
                tf.summary.histogram('1. Encoder F Space W', f_output, step=self.step // 128)

        return f_output

class Discriminator(tf.keras.layers.Layer):

    def __init__(self, conf_dict, tensorboard):
        super(Discriminator, self).__init__()
        self.latent_dim = conf_dict["LATENT_DIM"]
        self.z_dim = conf_dict["Z_DIM"]
        self.image_dim = conf_dict["IMAGE_DIM"]
        self.tensorboard = tensorboard
        self.step = 0
        self.input_layer = tf.keras.Input(self.latent_dim)
        self.l1 = tf.keras.layers.Dense(1024, input_dim = self.image_dim, activation=tf.keras.layers.LeakyReLU(alpha=0.2))
        self.l2 = tf.keras.layers.Dense(1024,  activation=tf.nn.relu)
        self.output_d = tf.keras.layers.Dense(1)
        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        # self.batch_norm_3 = tf.keras.layers.BatchNormalization()

        self.the_output = self.the_call(self.input_layer)

    def call(self, x, training=True):
        return self.the_call(x, training=training)

    def the_call(self, x, training=True):
        x = self.l1(x)
        x = self.batch_norm_1(x, training)
        x = self.l2(x)
        x = self.batch_norm_2(x, training)
        # x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
        disc_output = self.output_d(x)

        self.step += 1
        if self.tensorboard is not None and self.step % 20 == 0:
            with self.tensorboard.as_default():
                tf.summary.histogram('disc_output', disc_output, step=self.step * 10)

        return disc_output

# class E_encoder(tf.keras.Model):
class E_encoder(tf.keras.layers.Layer):

    def __init__(self, conf_dict, tensorboard):
        super(E_encoder, self).__init__()

        self.tensorboard = tensorboard
        self.step = 0
        self.latent_dim = conf_dict["LATENT_DIM"]
        self.z_dim = conf_dict["Z_DIM"]
        self.image_dim = conf_dict["IMAGE_DIM"]
        self.input_layer = tf.keras.Input(self.image_dim)
        self.conv_1 = tf.keras.layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same',
                                input_shape=[32, 32, 1],
                                kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02))
        self.conv_2 = tf.keras.layers.Conv2D(256, (4, 4), strides=(2, 2), padding='same',
                                kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02))
        self.conv_3 = tf.keras.layers.Conv2D(512, (4, 4), strides=(2, 2), padding='same',
                                kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02))
        self.conv_4 = tf.keras.layers.Conv2D(1, (4, 4), strides=(1, 1), padding='valid', kernel_initializer=tf.random_normal_initializer(mean=0., stddev=0.02))

        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        # self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        # self.batch_norm_3 = tf.keras.layers.BatchNormalization()

        self.flatten = tf.keras.layers.Flatten()
        self.fully_connected = tf.keras.layers.Dense(self.latent_dim) # No activation (pure linear)

        self.the_output = self.the_call(self.input_layer, training=False)

    def call(self, x, training=True):
        return self.the_call(x, training = training)

    def the_call(self, x, training=True):

        # An image 32x32 as input (source is generator or datum)
        # -1 is for batch size
        x = tf.reshape(x, (-1, 32, 32, 1)) # x = [None, 32, 32, 1]

        x = self.conv_1(x)
        x = self.batch_norm_1(x, training)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        # if enable_sa:
        #     x, _ = self.sa(x)

        x = self.conv_2(x)
        # x = self.batch_norm_2(x, training)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = self.conv_3(x)
        # x = self.batch_norm_3(x, training)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        x = self.flatten(x)
        x = self.fully_connected(x) # Linear

        self.step += 1
        if self.tensorboard is not None and self.step % 20 == 0:
            with self.tensorboard.as_default():
                tf.summary.histogram('2. Encoder E Space W', x, step=self.step // 128)

        return x # [-inf, +inf]

class Generator(tf.keras.layers.Layer):

    def __init__(self, conf_dict, tensorboard):
        super(Generator, self).__init__()

        self.latent_dim = conf_dict["LATENT_DIM"]
        self.input_layer = tf.keras.Input(self.latent_dim)
        self.tensorboard = tensorboard
        activation = None
        self.t1 = tf.keras.layers.Conv2DTranspose(512, (4, 4), strides=(1, 1), padding='valid', activation= activation, use_bias=False)
        self.t2 = tf.keras.layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same',  activation= activation,use_bias=False)
        self.t3 = tf.keras.layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same',  activation= activation,use_bias=False)
        self.t4 = tf.keras.layers.Conv2DTranspose(1, (4, 4), strides=(2, 2),   padding='same', use_bias=False)

        self.batch_norm_1 = tf.keras.layers.BatchNormalization()
        self.batch_norm_2 = tf.keras.layers.BatchNormalization()
        self.batch_norm_3 = tf.keras.layers.BatchNormalization()

        self.the_output = self.the_call(self.input_layer, training=False)

    def call(self, x, training=True):
        return self.the_call(x, training=training)

    def the_call(self, x, training=True):

        x = tf.reshape(x, (-1, 1, 1, self.latent_dim))

        # Up Sampling 1
        x = self.t1(x)
        x = self.batch_norm_1(x, training)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        # Up Sampling 2
        x = self.t2(x)
        x = self.batch_norm_2(x, training)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        # Up Sampling 3
        x = self.t3(x)
        x = self.batch_norm_3(x, training)
        x = tf.keras.layers.LeakyReLU(0.2)(x)

        # Last upsampling
        x = self.t4(x)

        return tf.reshape(x, (-1, 32 * 32)) # range is [-1, 1]


