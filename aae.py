from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, merge
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model

import matplotlib.pyplot as plt

import sys
import os

import numpy as np


class AAE():
    def __init__(self, result_path, load_data, rows, cols, chanels):
        self.img_rows = rows
        self.img_cols = cols
        self.channels = chanels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100
        self.kernel_size = 3
        self.result_path = result_path

        if not os.path.exists(result_path):
            os.mkdir(result_path)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        img = Input(shape=self.img_shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)

        # For the adversarial_autoencoder model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator determines validity of the encoding
        validity = self.discriminator(encoded_repr)

        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)


        # Load the dataset
        self.X_train, self.X_test = load_data()

        # Rescale -1 to 1
        self.X_train = self.X_train / 127.5 - 1.
        self.X_test = self.X_test / 127.5 - 1.

        self.plot_all_model()

    def build_encoder(self):
        # Encoder

        img = Input(shape=self.img_shape)

        h = Flatten()(img)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        mu = Dense(self.latent_dim)(h)
        log_var = Dense(self.latent_dim)(h)
        latent_repr = merge([mu, log_var],
                mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
                output_shape=lambda p: p[0])

        return Model(img, latent_repr, name='encoder')

    def build_decoder(self):


        input = Input(shape=(self.latent_dim,))
        z = Dense(512)(input)
        z = LeakyReLU(alpha=0.2)(z)
        z = Dense(512)(z)
        z = LeakyReLU(alpha=0.2)(z)
        z = Dense(np.prod(self.img_shape), activation='tanh')(z)
        img = Reshape(self.img_shape)(z)

        encoder = Model(input, img, name='decoder')
        encoder.summary()

        return encoder

    def build_discriminator(self):


        encoded_repr = Input(shape=(self.latent_dim, ))
        x = Dense(512)(encoded_repr)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dense(256)(x)
        x = LeakyReLU(alpha=0.2)(x)
        validity = Dense(1, activation="sigmoid")(x)

        discriminator = Model(encoded_repr, validity, name='discriminator')
        discriminator.summary()

        return discriminator

    def load_weight(self, dirpath):
        self.encoder.load_weights(os.path.join(dirpath, 'model_encoder.h5'))
        self.decoder.load_weights(os.path.join(dirpath, 'model_decoder.h5'))
        self.discriminator.load_weights(os.path.join(dirpath, 'model_discriminator.h5'))

    def save_weight(self, dirpath):
        self.encoder.save_weights(os.path.join(dirpath, 'model_encoder.h5'))
        self.decoder.save_weights(os.path.join(dirpath, 'model_decoder.h5'))
        self.discriminator.save_weights(os.path.join(dirpath, 'model_discriminator.h5'))

    def train(self, epochs, batch_size=128, sample_interval=50):

        X_train = self.X_train
        X_test = self.X_test

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))


        if not os.path.exists(os.path.join(self.result_path, 'epoch.txt')):
            start = 0
        else:
            with open(os.path.join(self.result_path, 'epoch.txt'), 'r') as file:
                start = int(file.readline())
            self.load_weight(self.result_path)
        end = start + epochs

        for epoch in range(start, end):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            latent_fake = self.encoder.predict(imgs)
            latent_real = np.random.normal(size=(batch_size, self.latent_dim))

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator
            g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [imgs, valid])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))
            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.reconstruct(epoch)
                self.save_weight(self.result_path)
                with open(os.path.join(self.result_path, 'epoch.txt'), 'w') as file:
                    file.truncate()
                    file.write(str(epoch))

        self.save_weight(self.result_path)
        with open(os.path.join(self.result_path, 'epoch.txt'), 'w') as file:
            file.truncate()
            file.write(str(epoch))

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.decoder.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        filename = '%d.png' % epoch
        plt.savefig(os.path.join(self.result_path, filename), dpi=300)
        plt.close()

    def reconstruct(self, epoch):
        X_train = self.X_train
        X_test = self.X_test

        test_no = 10
        original_image = X_test[:test_no]
        z_sample = self.encoder.predict(X_test[:test_no])
        reconstruct_image = self.decoder.predict(z_sample)

        # Rescale images 0 - 1
        original_image = 0.5 * original_image + 0.5
        reconstruct_image = 0.5 * reconstruct_image + 0.5

        figure = np.zeros((self.img_rows * 2, self.img_cols * 10, 3))
        for i in range(test_no):
            figure[0*self.img_rows:1*self.img_rows, i*self.img_cols:(i+1)*self.img_cols, :] = original_image[i]
            figure[1*self.img_rows:2*self.img_rows, i*self.img_cols:(i+1)*self.img_cols, :] = reconstruct_image[i]
        plt.figure()
        plt.title("reconstruct result")
        plt.imshow(figure)
        filename = 'reconstruct_result_%dep.png' % epoch
        plt.savefig(os.path.join(self.result_path, filename), dpi=300)
        plt.close()

    def plot_all_model(self):
        plot_model(self.encoder, to_file=os.path.join(self.result_path, 'encoder.png'), show_shapes=True)
        plot_model(self.decoder, to_file=os.path.join(self.result_path, 'decoder.png'), show_shapes=True)
        plot_model(self.discriminator, to_file=os.path.join(self.result_path, 'discriminator.png'), show_shapes=True)
        plot_model(self.adversarial_autoencoder, to_file=os.path.join(self.result_path, 'adversarial_autoencoder.png'), show_shapes=True)

    def hide_message(self, another_result_path):
        X_train = self.X_train
        X_test = self.X_test


        test_no = 10
        original_image = X_test[:test_no]

        self.load_weight(self.result_path)
        z_sample = self.encoder.predict(X_test[:test_no])
        direct_reconstruct_image = self.decoder.predict(z_sample)
        self.load_weight(another_result_path)
        coded_image = self.decoder.predict(z_sample)
        z_sample = self.encoder.predict(coded_image)
        self.load_weight(self.result_path)
        reconstruct_image = self.decoder.predict(z_sample)

        # Rescale images 0 - 1
        original_image = 0.5 * original_image + 0.5
        coded_image = 0.5 * coded_image + 0.5
        reconstruct_image = 0.5 * reconstruct_image + 0.5
        direct_reconstruct_image = 0.5 * direct_reconstruct_image + 0.5

        figure = np.zeros((self.img_rows * 4, self.img_cols * 10, 3))
        for i in range(test_no):
            figure[0*self.img_rows:1*self.img_rows, i*self.img_cols:(i+1)*self.img_cols, :] = original_image[i]
            figure[1*self.img_rows:2*self.img_rows, i*self.img_cols:(i+1)*self.img_cols, :] = direct_reconstruct_image[i]
            figure[2*self.img_rows:3*self.img_rows, i*self.img_cols:(i+1)*self.img_cols, :] = coded_image[i]
            figure[3*self.img_rows:4*self.img_rows, i*self.img_cols:(i+1)*self.img_cols, :] = reconstruct_image[i]
        plt.figure()
        plt.title("hide message result")
        plt.imshow(figure)
        filename = 'hide_message_result_%s_%s.png' % (self.result_path, another_result_path)
        plt.show()
        plt.savefig(os.path.join(self.result_path, filename), dpi=300)
        plt.close()



