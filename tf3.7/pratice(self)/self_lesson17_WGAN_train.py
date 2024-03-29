import os
import numpy as np
import tensorflow as tf
from PIL import Image
import glob
from practice_self.self_lesson17_WGAN import Generator, Discriminator
from practice.dataset import make_anime_dataset

def save_result(val_out, val_block_size, image_path, color_mode):
    def preprocess(img):
        img = ((img + 1.0) * 127.5).astype(np.uint8)
        # img = img.astype(np.uint8)
        return img

    preprocesed = preprocess(val_out)
    final_image = np.array([])
    single_row = np.array([])
    for b in range(val_out.shape[0]):
        # concat image into a row
        if single_row.size == 0:
            single_row = preprocesed[b, :, :, :]
        else:
            single_row = np.concatenate((single_row, preprocesed[b, :, :, :]), axis=1)

        # concat image row to final_image
        if (b+1) % val_block_size == 0:
            if final_image.size == 0:
                final_image = single_row
            else:
                final_image = np.concatenate((final_image, single_row), axis=0)

            # reset single row
            single_row = np.array([])

    if final_image.shape[2] == 1:
        final_image = np.squeeze(final_image, axis=2)
    Image.fromarray(final_image).save(image_path)

def celoss_zeros(d_fake_logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits))

    return tf.reduce_mean(loss)

def celoss_ones(d_real_logits):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits, labels=tf.ones_like(d_real_logits))

    return tf.reduce_mean(loss)

def gradient_penalty(discriminator, batch_x, fake_image):
    batchsz = batch_x.shape[0]
    t = tf.random.uniform([batchsz, 1, 1, 1])
    tf.broadcast_to(t, batch_x.shape)

    interplate = t * batch_x + (1-t) * fake_image

    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplate_logits = discriminator(interplate)
    grads = tape.gradient(d_interplate_logits, interplate)
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)
    gp = tf.reduce_mean((gp-1)**2)

    return gp

def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)

    d_loss_fake = celoss_zeros(d_fake_logits)
    d_loss_real = celoss_ones(d_real_logits)

    gp = gradient_penalty(discriminator, batch_x, fake_image)
    loss = d_loss_fake + d_loss_real + 10*gp

    return loss, gp

def g_loss_fn(generator, discriminator, batch_z, is_training):
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    loss = celoss_ones(d_fake_logits)

    return loss


def main():
    tf.random.set_seed(22)
    np.random.seed(22)

    z_dim = 100
    epochs = 300000
    batch_size = 256
    learning_rate = 0.0005
    is_training = True

    img_path = glob.glob(r'D:\pythonProject\tf3.7\practice\data\faces\*.jpg')
    dataset, img_shape, _ = make_anime_dataset(img_path, batch_size)
    dataset = dataset.repeat()
    db_iter = iter(dataset)

    generator = Generator()
    generator.build(input_shape=(None, z_dim))
    discriminator = Discriminator()
    discriminator.build(input_shape=(None, 64, 64, 3))

    g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

    for epoch in range(epochs):
        batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)
        batch_x = next(db_iter)

        with tf.GradientTape() as tape:
            d_loss, gp = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))

        with tf.GradientTape() as tape:
            g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

        if epoch % 100 == 0:
            print(epoch, 'd-loss:', float(d_loss), 'g-loss', float(g_loss),
                  'gp:', float(gp))

            z = tf.random.uniform([100, z_dim])
            fake_image = generator(z, training=False)
            img_path = os.path.join('images', 'wgan-%d.png' % epoch)
            save_result(fake_image.numpy(), 10, img_path, color_mode='P')

if __name__ == '__main__':
    main()
