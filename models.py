def vgg_block(input_vec, filters=32, l2_weight_regulaizer=0.0002, weight_initializer="he_uniform", kernel=(3, 3)):
    import tensorflow as tf
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
    from tensorflow.keras.regularizers import l2

    with tf.keras.backend.name_scope("VGG_Block"):
        conv3x3_1 = Conv2D(filters, kernel, padding='same', activation='relu', kernel_initializer=weight_initializer,
                           kernel_regularizer=l2(l2_weight_regulaizer))(input_vec)
        batch_norm_1 = BatchNormalization()(conv3x3_1)
        conv3x3_2 = Conv2D(filters, kernel, padding='same', activation='relu', kernel_initializer=weight_initializer,
                           kernel_regularizer=l2(l2_weight_regulaizer))(batch_norm_1)
        batch_norm_2 = BatchNormalization()(conv3x3_2)

        return MaxPooling2D(pool_size=2)(batch_norm_2)


def res_block(input_vec, filters=64, l2_weight_regulaizer=0.0002, weight_initializer="he_uniform", kernel=(3, 3)):
    import tensorflow as tf
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
    from tensorflow.keras.regularizers import l2

    with tf.keras.backend.name_scope("RES_Block"):

        conv3x3_1 = Conv2D(filters, kernel, padding='same', activation='relu', kernel_initializer=weight_initializer,
                           kernel_regularizer=l2(l2_weight_regulaizer))(input_vec)
        batch_norm_1 = BatchNormalization()(conv3x3_1)

        conv3x3_2 = Conv2D(filters, kernel, padding='same', activation='relu', kernel_initializer=weight_initializer,
                           kernel_regularizer=l2(l2_weight_regulaizer))(batch_norm_1)
        batch_norm_2 = BatchNormalization()(conv3x3_2)

        return MaxPooling2D(pool_size=2)(batch_norm_2)


def vgg_model(input_shape, l2_weight_regulaizer=0.0002, weight_initializer="he_uniform", num_classes=20):
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.regularizers import l2
    print("loading vgg_model as model..")
    inputs = tf.keras.layers.Input(input_shape[1:])

    vgg_blk1 = vgg_block(inputs, filters=32,  l2_weight_regulaizer=l2_weight_regulaizer,
                         weight_initializer=weight_initializer)

    vgg_blk2 = vgg_block(vgg_blk1, filters=64,  l2_weight_regulaizer=l2_weight_regulaizer,
                         weight_initializer=weight_initializer)
    vgg_blk3 = vgg_block(vgg_blk2, filters=128,  l2_weight_regulaizer=l2_weight_regulaizer,
                         weight_initializer=weight_initializer)

    vgg_blk_fin = vgg_block(vgg_blk3, filters=256,  l2_weight_regulaizer=l2_weight_regulaizer,
                            weight_initializer=weight_initializer)

    flatten_1 = Flatten()(vgg_blk_fin)
    dense_1 = Dense(256, activation='relu', kernel_initializer=weight_initializer,
                    kernel_regularizer=l2(0.0001))(flatten_1)
    drop_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(128, activation='relu', kernel_initializer=weight_initializer,
                    kernel_regularizer=l2(0.0001))(drop_1)
    drop_2 = Dropout(0.5)(dense_2)
    outputs = Dense(num_classes, activation='softmax')(drop_2)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


def bridged_vgg_model(input_shape, l2_weight_regulaizer=0.0002, weight_initializer="he_uniform", num_classes=20):
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Dropout, MaxPooling2D, Flatten, concatenate

    print("loading bridged_vgg_model as model..")
    inputs = tf.keras.layers.Input(input_shape[1:])

    vgg_blk1 = vgg_block(inputs, filters=32,  l2_weight_regulaizer=l2_weight_regulaizer,
                         weight_initializer=weight_initializer)

    vgg_blk2 = vgg_block(vgg_blk1, filters=64,  l2_weight_regulaizer=l2_weight_regulaizer,
                         weight_initializer=weight_initializer)
    vgg_blk3 = vgg_block(vgg_blk2, filters=128,  l2_weight_regulaizer=l2_weight_regulaizer,
                         weight_initializer=weight_initializer)

    vgg_blk_fin = vgg_block(vgg_blk3, filters=256,  l2_weight_regulaizer=l2_weight_regulaizer,
                            weight_initializer=weight_initializer)

    max2bridge = MaxPooling2D(pool_size=2)(vgg_blk2)
    merge_1 = concatenate([vgg_blk3, max2bridge], -1)
    vgg_blk1_1 = vgg_block(merge_1, filters=128,  l2_weight_regulaizer=l2_weight_regulaizer,
                           weight_initializer=weight_initializer, kernel=(1, 1))
    flatten_1 = Flatten()(vgg_blk_fin)
    flatten_2 = Flatten()(vgg_blk1_1)
    merge_2 = concatenate([flatten_1, flatten_2])
    dense_1 = Dense(512, activation='relu', kernel_initializer=weight_initializer)(merge_2)
    drop_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(128, activation='relu', kernel_initializer=weight_initializer)(drop_1)
    drop_2 = Dropout(0.5)(dense_2)
    outputs = Dense(num_classes, activation='softmax')(drop_2)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model


def model1(input_shape, l2_weight_regulaizer=0.0002, weight_initializer="he_uniform", num_classes=20):
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, BatchNormalization, concatenate
    from tensorflow.keras.regularizers import l2

    inputs = tf.keras.layers.Input(input_shape[1:])
    print("loading model1 as model")
    # 1st 3by3
    conv3_1 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer=weight_initializer,
                     kernel_regularizer=l2(l2_weight_regulaizer))(inputs)
    batch_1 = BatchNormalization()(conv3_1)
    conv3_2 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=weight_initializer,
                     kernel_regularizer=l2(l2_weight_regulaizer))(batch_1)
    batch_2 = BatchNormalization()(conv3_2)
    drop_1 = Dropout(0.4)(batch_2)
    max_1 = MaxPooling2D(pool_size=2)(drop_1)

    # 1st 1by1
    conv1_1 = Conv2D(62, (1, 1), activation='relu', kernel_initializer=weight_initializer,
                     kernel_regularizer=l2(l2_weight_regulaizer))(max_1)
    flat_1 = Flatten()(conv1_1)

    # 2nd 3by3
    conv3_3 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=weight_initializer,
                     kernel_regularizer=l2(l2_weight_regulaizer))(max_1)
    batch_2 = BatchNormalization()(conv3_3)
    conv3_4 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=weight_initializer,
                     kernel_regularizer=l2(l2_weight_regulaizer))(batch_2)
    batch_3 = BatchNormalization()(conv3_4)
    drop_2 = Dropout(0.5)(batch_3)
    max_2 = MaxPooling2D(pool_size=2)(drop_2)
    # Flat_2 = Flatten()(max_2)

    conv3_5 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=weight_initializer,
                     kernel_regularizer=l2(l2_weight_regulaizer))(max_2)
    batch_4 = BatchNormalization()(conv3_5)
    conv3_6 = Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer=weight_initializer,
                     kernel_regularizer=l2(l2_weight_regulaizer))(batch_4)
    batch_5 = BatchNormalization()(conv3_6)
    drop_4 = Dropout(0.5)(batch_5)
    max_3 = MaxPooling2D(pool_size=2)(drop_4)
    conv1_3 = Conv2D(32, (1, 1), activation='relu', kernel_initializer=weight_initializer,
                     kernel_regularizer=l2(l2_weight_regulaizer))(max_3)
    flat_2 = Flatten()(conv1_3)
    # 2nd 1by1

    conv1_2 = Conv2D(16, (1, 1), activation='relu', kernel_initializer=weight_initializer,
                     kernel_regularizer=l2(l2_weight_regulaizer))(max_2)
    flat_3 = Flatten()(conv1_2)

    # merge
    merge = concatenate([flat_1, flat_2, flat_3])
    dense_1 = Dense(128, activation='relu', kernel_initializer=weight_initializer,
                    kernel_regularizer=l2(0.0001))(merge)
    drop_3 = Dropout(0.5)(dense_1)

    outputs = Dense(num_classes, activation='softmax')(drop_3)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.summary()
    return model
