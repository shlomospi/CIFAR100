import numpy as np
import utilsCIFAR100 as utilCIFAR
import models as loc_models
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras
import os
from datetime import datetime
from trains import Task, Logger

# TODO change dir to generalized working dir
# TODO github integration
task = Task.init()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Parameters -----------------------------------------------------------------------------------------------------------

Epochs = 150
validation_part = 0.1
LearningRate = 0.001
weight_initializer = "he_uniform"
l2_weight_regulaizer = 0.0004
image_shape = (32, 32)
Channels = 3
NumClasses = 20
Batch_Size = 32
show_img_after = False
optimizer = "ADAM"

if optimizer == "ADAM":
    optimizer = tf.keras.optimizers.Adam(learning_rate=LearningRate)
elif optimizer == "SGD":
    optimizer = tf.keras.optimizers.SGD(learning_rate=LearningRate, momentum=0.9)


config = "BatchSize_{}_Epochs_{}_LearningRate_{}".format(Batch_Size, Epochs, str(LearningRate)[2:])
date = datetime.now().strftime("%d%m%Y_%H%M")
log_folder = "log/{}".format(date+"_"+config)
bashCommand_tensorboard = "tensorboard --logdir=/projects/CIFAR100_Functional/{}".format(log_folder)
os.makedirs(log_folder)
print("TensorBoard results page: run:\ntensorboard --logdir=/projects/CIFAR100_Functional/{}".format(log_folder))
saved_file_name = '/tmp/model_best_{epoch:02d}-{val_loss:.2f}'
part_name = "coarse"

# Unpack data-----------------------------------------------------------------------------------------------------------

x_train, x_test, y_train_by_num, y_test_by_num, fine_names, coarse_names = utilCIFAR.unpack_cifar100(paint=False)

y_test = tf.keras.utils.to_categorical(np.asarray(y_test_by_num))
y_train = tf.keras.utils.to_categorical(np.asarray(y_train_by_num))

x_test = np.asarray(x_test)
x_train = np.asarray(x_train)
print("Split validation from train..")
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_part, random_state=7)

# -----------------------------Data Augmentation------------------------------------------------------------------------
print("Add flipped augmentation to x_train..")
x_train, y_train = utilCIFAR.horizontal_flip_and_show(x_train, y_train, verbose=1)

print("TrainX shape is {}. ValidationX shape is {}. TestX shape is {}".format(x_train.shape, x_val.shape, x_test.shape))
print("TrainY shape is {}.        ValidationY shape is {}.        TestY shape is {}".format(y_train.shape,
                                                                                            y_val.shape, y_test.shape))

# ----------------------------Data Normalization------------------------------------------------------------------------

x_train = utilCIFAR.standardize(x_train, tilting=0, normalization=255.0)
x_test = utilCIFAR.standardize(x_test, tilting=0, normalization=255.0)
x_val = utilCIFAR.standardize(x_val, tilting=0, normalization=255.0)

# -----------------------------------Build model------------------------------------------------------------------------

input_shape = (Batch_Size, image_shape[0], image_shape[1], Channels)
inputs = tf.keras.layers.Input(input_shape[1:])
model = loc_models.bridged_vgg_model(input_shape=input_shape, l2_weight_regulaizer=l2_weight_regulaizer,
                                     weight_initializer=weight_initializer, num_classes=20)

# ---------------------------------Compile model------------------------------------------------------------------------

model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer=optimizer,
              metrics=[tf.keras.metrics.CategoricalAccuracy()]
              )

# -------------------------------------Callbacks------------------------------------------------------------------------

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=log_folder, verbose=1, monitor='val_categorical_accuracy',
                                                  mode='max', save_best_only=True)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir=log_folder, histogram_freq=1, write_images=True, write_graph=False)

adaptivelr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.15, patience=2,
                                                  verbose=0, mode='auto', cooldown=2, min_lr=0.00001)
Logger.report_scalar(title="learning rate", value=LearningRate)
Earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=25,
                                                 verbose=0, mode='auto', baseline=None, restore_best_weights=True)

# Show image after prep-------------------------------------------------------------------------------------------------

if show_img_after:
    from PIL import Image
    PIL_image = Image.fromarray(np.uint8((x_train[4, :, :, :])*255.0)).convert('RGB')
    PIL_image.show()
print("Train initialized, follow by executing:")
print(r"tensorboard --logdir=/home/shlomo/Projects/CIFAR100_Functional/{}".format(log_folder))
train_loss, train_score = model.evaluate(x_train, y_train, verbose=0)
validation_loss, validation_score = model.evaluate(x_val, y_val, verbose=0)
test_loss, test_score = model.evaluate(x_test, y_test, verbose=0)
print('Train accuracy:     %.3f' % (train_score*100) + " train loss: " + str(train_loss))
print('validation accuracy: %.3f' % (validation_score*100) + " validation loss: " + str(validation_loss))
print('Test accuracy:      %.3f' % (test_score*100) + "   test loss: " + str(test_loss))

# Train model-----------------------------------------------------------------------------------------------------------

history = model.fit(x_train,
                    y_train,
                    batch_size=Batch_Size,
                    epochs=Epochs,
                    validation_data=(x_val, y_val),  # (x_test,y_test), #
                    verbose=2,
                    callbacks=[checkpointer, tensorboard, Earlystopping])  # adaptivelr,

print(history.history)

# Load the weights with the best validation accuracy
# model.load_weights(saved_file_name)
print("Saving")
model.save('saved_model_'+part_name)
print("Loading")
model = keras.models.load_model('saved_model_'+part_name)
# Evaluate the model on test set
_, train_score = model.evaluate(x_train, y_train, verbose=0)
_, validation_score = model.evaluate(x_val, y_val, verbose=0)
_, test_score = model.evaluate(x_test, y_test, verbose=0)
Result = model.predict(x_test)

#  confusion matrix
Result_by_num = tf.argmax(Result, axis=1)
confusion_matrix = tf.math.confusion_matrix(labels=y_test_by_num, predictions=Result_by_num)
utilCIFAR.plot_confusion_matrix(confusion_matrix.numpy(), coarse_names, normalize=True)

#  Print test accuracy

print('Train accuracy: %.3f' % (train_score*100))
print('validation accuracy: %.3f' % (validation_score*100))
print('Test accuracy: %.3f' % (test_score*100))
print(history.history.keys())
utilCIFAR.plot_acc_lss(history)
