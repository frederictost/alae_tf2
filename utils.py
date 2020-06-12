import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf

def tf_process_images(image):

  img_shaped =  tf.dtypes.cast(tf.reshape(image, [-1, 28, 28, 1]), tf.float32)
  img_shaped_resized = tf.image.resize(img_shaped, [32, 32])
  img_shaped_resized = (img_shaped_resized / 127.5) - 1

  return img_shaped_resized

def plot_mnist_grid(samples, target_file = None):

    # 8 x 8 grid
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(32, 32), cmap='gray')

    if target_file is not None:
        plt.savefig(target_file, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig