
"""
Script pour définir l'architecture du générateur et du discriminateur

"""


import math
import tensorflow as tf


# 1- construction du générateur

# encodeur

def downsample(filters, size, apply_batchnorm=False):
    """
    _Summary_ : fonction qui construit une couche de downsampling (encodeur)
    _Args_ :  filters (nombre de filtres),
              size (taille du kernel),
              apply_batchnorm (booléen pour appliquer ou non la batchnorm)
    _Returns_ : la couche de downsampling
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result


# decodeur

def upsample(filters, size, apply_dropout=True):
    """
    _Summary_ : fonction qui construit une couche de upsampling (decodeur)
    _Args_ :  filters (nombre de filtres),
              size (taille du kernel),
              apply_dropout (booléen pour appliquer ou non le dropout)
    _Returns_ : la couche de upsampling
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                        padding='same',
                                        kernel_initializer=initializer,
                                        use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.LeakyReLU())
    return result


# generateur = encodeur + decodeur

def Generator(image_size=512, stride=2, kernel=4, OUTPUT_CHANNELS=1):
    """
    _Summary_ : fonction qui construit le générateur U-Net
    _Args_ :  image_size (taille de l'image d'entrée),
              stride (stride des convolutions),
              kernel (taille du kernel),
              OUTPUT_CHANNELS (nombre de canaux de sortie)
    _Returns_ : le modèle du générateur
    """
    num_layers = math.ceil(math.log(image_size) / math.log(stride))
    inputs = tf.keras.layers.Input(
        shape=[image_size, image_size, OUTPUT_CHANNELS])
    down_stack = []
    up_stack = []
    filters = 64
    for i in range(num_layers):
        # applique la batchnorm à toutes les couches exceptée la première : à changer ??
        down_stack.append(downsample(
            filters, kernel, apply_batchnorm=(i != 0)))
        filters = min(filters * 2, 512)  # Bloque les filtres à 512 max
    for i in range(num_layers-1):
        up_stack.append(upsample(filters, kernel, apply_dropout=(i != 0)))
        filters = max(filters // 2, 64)
    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation="tanh")
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = list(reversed(skips[:-1]))
    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])
    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)


# generator = Generator(256,2,4,3)  # test
# affichage de l'architecture du generateur
# tf.keras.utils.plot_model(generator, to_file="generator_model.png", show_shapes=True, dpi=64)


# 2- construction du discriminateur

def Discriminator(image_size=256, stride=1, kernel=4, OUTPUT_CHANNELS=3):
    """
    _Summary_ : fonction qui construit le discriminateur PatchGAN
    _Args_ :  image_size (taille de l'image d'entrée),
              stride (stride des convolutions),
              kernel (taille du kernel),
              OUTPUT_CHANNELS (nombre de canaux de sortie)
    _Returns_ : le modèle du discriminateur
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    inp = tf.keras.layers.Input(shape=[image_size, image_size, OUTPUT_CHANNELS], name='input_image')
    tar = tf.keras.layers.Input(shape=[image_size, image_size, OUTPUT_CHANNELS], name='target_image')
    x = tf.keras.layers.concatenate([inp, tar])
    down1 = downsample(64, kernel, False)(x)  
    down2 = downsample(128, kernel)(down1)  
    down3 = downsample(256, kernel)(down2)  
    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  
    conv = tf.keras.layers.Conv2D(512, kernel, strides=stride,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1) 
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    zero_pad2 = tf.keras.layers.ZeroPadding2D()(
        leaky_relu)  
    last = tf.keras.layers.Conv2D(1, kernel, strides=stride,
                                  kernel_initializer=initializer)(zero_pad2)
    return tf.keras.Model(inputs=[inp, tar], outputs=last)


# discriminator = Discriminator()  # test
# affichage de l'architecture du discriminateur
# tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

