# Standard modules
import os
import datetime
import numpy as np
# Tensorflow
import tensorflow as tf
from tensorflow.keras import datasets
from tensorboard.plugins.hparams import api as hp
# In house module
import alae_tf2_helper as alae
import utils
import alae_tf2_models

def main():

    print("Tensorflow version {}".format(tf.__version__))

    # Configuration
    MODEL_NAME    = "ALAE_CONV_V1"
    generate_mnist_samples = False
    generate_samples_tensorboard = True
    PRINT_IT  = 50
    RESULT_IT = 100
    SAVE_WEIGHT_IT = 5000

    # Network configuration & hyper parameters
    EPOCHS       = 100
    BATCH_SIZE   = 128
    Z_DIM        = 100
    LATENT_DIM   = 50
    GAMMA_GP     = 10
    K_RECONST_KL = 0.5 # Latent space quality, Pure reconstruction & Kullback Leibler ratio
    # Learning Rate for Discriminator, Generator, Latent Space
    LR_D_G_L     = [0.0001,0.0004,0.0002] # Best

    #
    # Manage folders
    #
    original_mnist_samples = os.path.join("results", "mnist_original")
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    mnist_samples_ = os.path.join("results", MODEL_NAME + "_" + current_time)
    folder_to_create = [original_mnist_samples,mnist_samples_]
    # Create folders
    for folder in folder_to_create:
        if not os.path.exists(folder):
            os.makedirs(folder)
    # Define folders
    checkpoint_path        = os.path.join("checkpoint", MODEL_NAME)
    original_mnist_samples = os.path.join(original_mnist_samples, "samples_{}.png")
    mnist_samples = os.path.join(mnist_samples_, "alae_samples_{}.png")
    static_mnist_samples = os.path.join(mnist_samples_, "static_alae_samples_{}.png")
    train_log_dir = os.path.join("logs", "tensorboard", MODEL_NAME + "_" + current_time)
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    # Log hyper parameters
    HP_BATCH_SIZE = hp.HParam('HP_BATCH_SIZE', hp.Discrete([64,128,256,512,1024]))
    HP_Z_DIM      = hp.HParam('HP_Z_DIM',      hp.Discrete([50,100,200]))
    HP_LATENT_DIM = hp.HParam('HP_LATENT_DIM', hp.Discrete([30,50,70]))
    HP_GAMMA_GP   = hp.HParam('HP_GAMMA_GP',   hp.Discrete([2,5,10]))
    HP_K_RECONST_KL     = hp.HParam('HP_K_RECONST_KL',     hp.RealInterval (0.,1.))
    HP_LR_GENERATOR     = hp.HParam('HP_LR_GENERATOR',     hp.RealInterval (0.,0.1))
    HP_LR_DISCRIMINATOR = hp.HParam('HP_LR_DISCRIMINATOR', hp.RealInterval (0.,0.1))
    HP_LR_LATENT        = hp.HParam('HP_LR_LATENT',        hp.RealInterval (0.,0.1))

    hparams = {
        HP_BATCH_SIZE:       BATCH_SIZE,
        HP_Z_DIM:            Z_DIM,
        HP_LATENT_DIM:       LATENT_DIM,
        HP_GAMMA_GP:         GAMMA_GP,
        HP_K_RECONST_KL:     K_RECONST_KL,
        HP_LR_DISCRIMINATOR: LR_D_G_L[0],
        HP_LR_GENERATOR:     LR_D_G_L[1],
        HP_LR_LATENT:        LR_D_G_L[2],
    }

    # Log hyper parameters
    METRIC_LATENT_LOST = 'latent_loss'
    with train_summary_writer.as_default():
        hp.hparams_config(hparams, metrics=[hp.Metric(METRIC_LATENT_LOST, display_name='Latent Loss')])
        hp.hparams(hparams, trial_id = MODEL_NAME + "_" + current_time)
        tf.summary.scalar(METRIC_LATENT_LOST, 0, step=1)


    # Do useful stuff
    seed = 2020
    np.random.seed(seed)
    tf.random.set_seed(seed)

    #
    # Load data
    #
    (x_train, _), (_, _) = datasets.mnist.load_data()

    # Prepare the data
    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    train_dataset = train_dataset.batch(BATCH_SIZE)
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.map(utils.tf_process_images)

    # x_train = utils.process_images(x_train)
    IMAGE_DIM = 32 * 32
    # x_train = tf.dtypes.cast(tf.reshape(x_train, (len(x_train), IMAGE_DIM) ), tf.float32)

    # A way to plot some real examples of MNIST
    if generate_mnist_samples:
        for index in range(16):
            source_indexes = np.random.permutation(len(x_train.numpy()))[0:64]
            utils.plot_mnist_grid(x_train.numpy()[source_indexes], target_file = original_mnist_samples.format(index))

    #
    # Prepare the models
    #
    conf_dict = {"Z_DIM": Z_DIM, "LATENT_DIM": LATENT_DIM,
                 "IMAGE_DIM": IMAGE_DIM,
                 "GAMMA_GP": GAMMA_GP,
                 "LR_D_G_L": LR_D_G_L,
                 "K_RECONST_KL": K_RECONST_KL, }

    generator       = alae_tf2_models.Generator(conf_dict,train_summary_writer)
    discriminator   = alae_tf2_models.Discriminator(conf_dict,train_summary_writer)
    E_encoder       = alae_tf2_models.E_encoder(conf_dict,train_summary_writer)
    F_encoder       = alae_tf2_models.F_encoder(conf_dict,train_summary_writer)

    alae_helper = alae.alae_helper({"generator":generator,
                                    "discriminator":discriminator,
                                    "E_encoder":E_encoder,
                                    "F_encoder":F_encoder,}, conf_dict)

    # Prepate to save the weights
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator,
                                     discriminator_optimizer=discriminator,
                                     E_encoder_optimizer=E_encoder,
                                     F_encoder_optimizer=F_encoder,
                                     generator=generator,
                                     discriminator=discriminator,
                                     E_encoder=E_encoder,
                                     F_encoder=F_encoder,
                                     step=tf.Variable(1))

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)

    z_samples_static = alae_helper.sample_Z(64, Z_DIM)

    # ------------------------------------------------------------------------------------
    # Start of the training loop
    # ------------------------------------------------------------------------------------
    it = 1
    for epoch in range(EPOCHS):

        for x in train_dataset:

            if it % RESULT_IT == 0:
                # 8x8 = 64 as a grid containing figures 0 to 9
                z_samples = alae_helper.sample_Z(64, Z_DIM)
                samples = generator( F_encoder(z_samples, training=False), training = False)
                img = samples.numpy()
                utils.plot_mnist_grid(img, target_file=mnist_samples.format(str(it).zfill(3)))

                samples = generator(F_encoder(z_samples_static, training=False), training=False)
                img = samples.numpy()
                utils.plot_mnist_grid(img, target_file=static_mnist_samples.format(str(it).zfill(3)))

                if generate_samples_tensorboard:
                    # Add results into Tensorboard
                    # (batch_size, height, width, channels)
                    img = np.reshape(img, [64,32,32,1] )
                    with train_summary_writer.as_default():
                        tf.summary.image("Generated Image", img, step=it)

            # The job is done here, x are real samples
            losses = alae_helper.trainstep(x)

            if it % PRINT_IT == 0:
                print('Epoch: {}   it: {}     L_loss: {:.4f}     D_loss: {:.4f}     G_loss: {:.4f}'.format(1 + (it * BATCH_SIZE) // x_train.shape[0],
                                                                                                           it, losses["latent"], losses["disc"], losses["gen"]))
                with train_summary_writer.as_default():

                    tf.summary.scalar('discriminator_loss', losses["disc"], step=it)
                    tf.summary.scalar('generator_loss', losses["gen"], step=it)
                    tf.summary.scalar('latent_loss', losses["latent"], step=it)
                    tf.summary.scalar('latent_loss_reconst', losses["latent_reconst"], step=it)
                    tf.summary.scalar('latent_loss_kl', losses["latent_kl"], step=it)
                    tf.summary.histogram('real_samples', x, step=it)

            if it % SAVE_WEIGHT_IT == 0:
                # Save the weights
                checkpoint.step.assign_add(1)
                save_path = manager.save()
                print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))

            it+=1 # Next iteration

if __name__ == "__main__":
    main()