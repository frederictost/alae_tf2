import tensorflow as tf

class alae_helper(tf.keras.Model):

    def __init__(self, models, conf_dict):
        super(alae_helper, self).__init__()

        # Get configuration
        self.z_dim      = conf_dict["Z_DIM"]
        self.latent_dim = conf_dict["LATENT_DIM"]
        self.image_dim  = conf_dict["IMAGE_DIM"]
        self.gamma      = conf_dict["GAMMA_GP"]
        self.lr_D_G_L   = conf_dict["LR_D_G_L"]
        self.k_reconst_kl = conf_dict["K_RECONST_KL"]

        # Define the models D,E,F,G
        self.generator     = models["generator"]
        self.discriminator = models["discriminator"]
        self.E_encoder     = models["E_encoder"]
        self.F_encoder     = models["F_encoder"]

        # Build the model giving the input size as a tuple
        self.generator.build((None, self.latent_dim))
        self.discriminator.build((None, self.latent_dim))
        self.E_encoder.build((None, self.image_dim))
        self.F_encoder.build((None, self.z_dim))

        # Prepare the sequences for loss assessment
        self.fakepass   = tf.keras.Sequential([self.F_encoder, self.generator, self.E_encoder, self.discriminator])
        self.realpass   = tf.keras.Sequential([self.E_encoder, self.discriminator])
        self.latentpass = tf.keras.Sequential([self.F_encoder, self.generator, self.E_encoder])

        # Define trainable variables
        self.E_D_var = self.E_encoder.trainable_variables + self.discriminator.trainable_variables
        self.F_G_var = self.F_encoder.trainable_variables + self.generator.trainable_variables
        self.G_E_var = self.generator.trainable_variables + self.E_encoder.trainable_variables

        # Optimizers
        # self.E_D_opt = tf.keras.optimizers.Adam(0.0001, beta_1=0.0, beta_2=0.99)
        # self.F_G_opt = tf.keras.optimizers.Adam(0.0004, beta_1=0.0, beta_2=0.99)
        # self.E_G_opt = tf.keras.optimizers.Adam(0.0002, beta_1=0.0, beta_2=0.99)

        self.E_D_opt = tf.keras.optimizers.Adam(self.lr_D_G_L[0], beta_1=0.0, beta_2=0.99)
        self.F_G_opt = tf.keras.optimizers.Adam(self.lr_D_G_L[1], beta_1=0.0, beta_2=0.99)
        self.E_G_opt = tf.keras.optimizers.Adam(self.lr_D_G_L[2], beta_1=0.0, beta_2=0.99)

    @tf.function
    def _disc_loss(self, z, x):

        fakeloss = tf.reduce_mean(tf.math.softplus(self.fakepass(z)))
        realloss = tf.reduce_mean(tf.math.softplus(-self.realpass(x)))

        with tf.GradientTape() as tape:
            tape.watch(x)
            realloss_2 = tf.reduce_sum(self.realpass(x))

        grad = tape.gradient(realloss_2, x)
        gradreg = self.gamma / 2 * tf.reduce_mean(tf.reduce_sum(tf.square(grad), axis = 1))

        return fakeloss + realloss + gradreg

    @tf.function
    def _gen_loss(self, z, _=None):
        return tf.reduce_mean(tf.math.softplus(-self.fakepass(z)))

    @tf.function
    def _latent_loss(self, z, _=None):
        # From z to w through F mapper
        latent = self.F_encoder(z)
        # From z to w through F, w, G, E
        recovered = self.latentpass(z)

        # Latent space w should be equals in distribution
        kl = tf.keras.losses.KLDivergence()
        loss_reconst = tf.reduce_mean(tf.square(latent - recovered))
        loss_kl      = kl(tf.math.sigmoid(latent), tf.math.sigmoid(recovered))

        return self.k_reconst_kl * loss_reconst + (1 - self.k_reconst_kl) * loss_kl, loss_reconst, loss_kl
        # return tf.reduce_mean(tf.square(latent - recovered))

    # @tf.function
    """def _latent_loss(self, z, x):
        # From z to w through F mapper
        latent = self.F_encoder(z)
        # From z to w through F, w, G, E
        recovered = self.latentpass(z)

        # k * | | x - GoE(x) | |²
        x = tf.keras.layers.Flatten()(x)
        w_space = self.E_encoder(x)
        x_hat = self.generator(w_space)
        reconst =  tf.reduce_mean(tf.square(x - x_hat))

        k = 0.5
        dist = tf.reduce_mean(tf.square(latent - recovered))
        return k * dist + (1-k) * reconst"""

    def _update(self, x, loss_fn, var, opt):
        z = self.sample_Z( x.shape[0], self.z_dim)

        with tf.GradientTape() as tape:
            loss = loss_fn(z, x)

        grad = tape.gradient(loss, var)
        opt.apply_gradients(zip(grad, var))

        return z.numpy(), loss.numpy()

    def _update_latent_loss(self, x, loss_fn, var, opt):
        z = self.sample_Z(x.shape[0], self.z_dim)

        with tf.GradientTape() as tape:
            loss, loss_reconst, loss_kl = loss_fn(z, x)

        grad = tape.gradient(loss, var)
        opt.apply_gradients(zip(grad, var))

        return loss.numpy(), loss_reconst.numpy(), loss_kl.numpy()

    def trainstep(self, x):
        _, dloss = self._update(x, self._disc_loss,   self.E_D_var, self.E_D_opt)
        _, gloss = self._update(x, self._gen_loss,    self.F_G_var, self.F_G_opt)
        # _, lloss = self._update(x, self._latent_loss, self.G_E_var, self.E_G_opt)
        lloss, lloss_reconst, lloss_kl = self._update_latent_loss(x, self._latent_loss, self.G_E_var, self.E_G_opt)

        return {
            "disc":   dloss,
            "gen":    gloss,
            "latent": lloss,
            "latent_reconst": lloss_reconst,
            "latent_kl": lloss_kl
        }

    def sample_Z(self, batch, dim):
        return tf.random.normal((batch, dim), 0, 1)
        # return np.random.uniform(-1., 1., size=[m, n])
