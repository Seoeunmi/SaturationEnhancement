import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, LeakyReLU, PReLU, Flatten, Dense


class UNet(tf.keras.Model):
    def __init__(self, channel=16, dtype='float32'):
        super(UNet, self).__init__()
        tf.keras.backend.set_floatx(dtype)

        self.input_conv = Conv1D(channel, 1, strides=1, padding='same', activation='tanh')

        self.encoder = []
        self.encoder.append(Conv1D(channel, 7, strides=2, padding='same', activation='relu'))
        self.encoder.append(Conv1D(channel, 7, strides=2, padding='same', activation='relu'))
        self.encoder.append(Conv1D(channel, 7, strides=2, padding='same', activation='relu'))
        self.encoder.append(Conv1D(channel, 7, strides=2, padding='same', activation='relu'))

        self.decoder = []
        self.decoder.append(Conv1DTranspose(channel, 7, strides=2, padding='same', activation='relu'))
        self.decoder.append(Conv1DTranspose(channel, 7, strides=2, padding='same', activation='relu'))
        self.decoder.append(Conv1DTranspose(channel, 7, strides=2, padding='same', activation='relu'))
        self.decoder.append(Conv1DTranspose(channel, 7, strides=2, padding='same', activation='relu'))

        self.output_conv = Conv1DTranspose(1, 1, strides=1, padding='same', activation='tanh')


    def call(self, x):
        x_input = self.input_conv(x)
        x_enc1 = self.encoder[0](x_input)
        x_enc2 = self.encoder[1](x_enc1)
        x_enc3 = self.encoder[2](x_enc2)
        x_enc4 = self.encoder[3](x_enc3)

        x_dec4 = self.decoder[0](x_enc4)
        x_dec3 = self.decoder[1](tf.concat([x_dec4, x_enc3], axis=2))
        x_dec2 = self.decoder[2](tf.concat([x_dec3, x_enc2], axis=2))
        x_dec1 = self.decoder[3](tf.concat([x_dec2, x_enc1], axis=2))
        x_output = self.output_conv(x_dec1)

        return x_output



class Detection(tf.keras.Model):
    def __init__(self, dtype='float32'):
        super(Detection, self).__init__()
        tf.keras.backend.set_floatx(dtype)

        self.conv1 = Conv1D(4, 7, strides=2, padding='same', activation='relu')
        self.conv2 = Conv1D(4, 7, strides=2, padding='same', activation='relu')
        self.conv3 = Conv1D(1, 7, strides=2, padding='same', activation='relu')
        self.flatten = Flatten()
        self.dnn = Dense(1, activation='sigmoid')


    def call(self, x, y):
        out = self.conv1(tf.concat([x, y], axis=2))
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten(out)
        out = self.dnn(out)

        return out


class Detection_before(tf.keras.Model):
    def __init__(self, dtype='float32'):
        super(Detection_before, self).__init__()
        tf.keras.backend.set_floatx(dtype)

        self.conv1 = Conv1D(4, 7, strides=2, padding='same', activation='relu')
        self.conv2 = Conv1D(4, 7, strides=2, padding='same', activation='relu')
        self.conv3 = Conv1D(1, 7, strides=2, padding='same', activation='relu')
        self.flatten = Flatten()
        self.dnn = Dense(1, activation='sigmoid')


    def call(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.flatten(out)
        out = self.dnn(out)

        return out
