

def unpickle(file):
    """
    create a dictionary from the original cifar train test and meta files
    from the original site, https://www.cs.toronto.edu/~kriz/cifar.html:"
    Loaded in this way, each of the batch files contains a dictionary with the following elements:
    data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image.
    The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
    The image is stored in row-major
    order, so that the first 32 entries of the array are the red channel values of the first row of the image.
    labels -- a list of 10000 numbers in the range 0-9.
    The number at index i indicates the label of the ith image in the array data.

    The dataset contains another file, called batches.meta. It too contains a Python dictionary object.
    It has the following entries:
    label_names -- a 10-element list which gives meaningful names to the numeric labels
    in the labels array described above.
     For example, label_names[0] == "airplane", label_names[1] == "automobile", etc."

    """
    import pickle
    with open(file, 'rb') as fo:
        dictionary = pickle.load(fo, encoding='bytes')
    return dictionary


def reshape_by_channels(array, height=32, width=32, channels=3):
    """
    Takes an array of vectors and reshapes them to the 2D picture
    :param array: Array of the data, each line a picture
    :param height: Target size
    :param width: Target size
    :param channels: Target size
    :return: Tensor of 2D pictures
    """
    import numpy as np
    assert array.shape[1] == height*width*channels
    images = list()
    if channels == 3:
        for entry in array:
            image = np.zeros((height, width, 3), dtype=np.uint8)
            for channel in range(channels):
                image[..., channel] = np.reshape(entry[channel*height*width:(channel+1)*height*width], (height, width))
            images.append(image)
    elif channels == 1:
        for entry in array:
            image = np.zeros((height, width, 1), dtype=np.uint8)
            image[..., 0] = np.reshape(entry, (height, width))  # Red channel
    else:
        raise()
    return images


def tensor_to_folder(tensor, filenames, coarse_labels, coarse_label_names, fine_labels, fine_label_names,
                     imgdir="image", listdir="data/cifar100.csv"):
    """
    convert tensor of pixel values, a list of labels and a list of labels to image folder with a csv list
    :param tensor: tensor of (image,height,width,channels)
    :param filenames: list of names for pictures
    :param coarse_labels: the coarse labels, by number
    :param coarse_label_names: list of named coarse labels
    :param fine_labels: the fine labels, by number
    :param fine_label_names: list of named fine labels
    :param imgdir: dir for image creation
    :param listdir: dir for list creation
    :return: none
    """
    from tqdm import tqdm
    from PIL import Image
    from os import stat
    with open(listdir, 'a', newline='') as CSVdatafile:
        if stat(listdir).st_size is 0:
            CSVdatafile.write("Path and name, coarse name, fine name\n")
        for index, image in tqdm(enumerate(tensor)):
            filename = filenames[index]
            label_c = coarse_label_names[coarse_labels[index]]
            label_f = fine_label_names[fine_labels[index]]
            pic = Image.fromarray(image)
            image_rgb = pic.convert('RGB')  # color image
            image_rgb.save(imgdir+"/"+filename, format="PNG")    # create the pics
            CSVdatafile.write(imgdir+"/"+filename+","+label_c+","+label_f+"\n")  # create a list of pics


def unpack_cifar100(paint=True, verbose=0):
    """
    prepare the CIFAR100 file into workable tensors.

    :param verbose: 0 for quite, 1 for updates
    :param paint: if True, create folders, pictures and a list of all pictures in the working dir.
    :return: xtrain, xtest, ytrain, ytest
    """
    meta = unpickle('meta')
    if verbose == 1:
        print("Opened meta.  keys are: ", meta.keys())
    fine_label_names = [t.decode('utf8') for t in meta[b'fine_label_names']]  # List of decoded labels
    coarse_label_names = [t.decode('utf8') for t in meta[b'coarse_label_names']]  # List of decoded labels

    train = unpickle('train')
    if verbose == 1:
        print("Opened train. keys are: ", train.keys())
    train_tensor = reshape_by_channels(train[b'data'])
    if paint:
        tensor_to_folder(tensor=train_tensor,
                         filenames=[t.decode('utf8') for t in train[b'filenames']],
                         coarse_labels=train[b'coarse_labels'],
                         coarse_label_names=coarse_label_names,
                         fine_labels=train[b'fine_labels'],
                         fine_label_names=fine_label_names,
                         imgdir="image/train",
                         listdir="data/CIFAR100.csv")

    test = unpickle('test')
    if verbose == 1:
        print("Opened test.  keys are:  {}".format(test.keys()))
    test_tensor = reshape_by_channels(test[b'data'])
    if paint:
        tensor_to_folder(tensor=test_tensor,
                         filenames=[t.decode('utf8') for t in test[b'filenames']],
                         coarse_labels=test[b'coarse_labels'],
                         coarse_label_names=coarse_label_names,
                         fine_labels=test[b'fine_labels'],
                         fine_label_names=fine_label_names,
                         imgdir="image/test",
                         listdir="data/CIFAR100.csv")

    return train_tensor, test_tensor, train[b'coarse_labels'], test[b'coarse_labels'],\
        fine_label_names, coarse_label_names


def standardize(x, tilting=0, normalization=1):
    """
    shift data tensor and divide
    :param x: Data tensor
    :param tilting: the shift.
                    if eq. 'mean', will calculate mean of all pictures.
                    if eq. 'personal mean'', will calculate mean per picture
    :param normalization: the denumorator.
                           if eq. 'STD' will divide by the STD of all pictures.
                           if eq. 'personal STD' will divide by the STD of each picture.
    :return: the data after
    """
    import numpy as np
    x = np.asarray(x)
    x = x.astype('float32')
    if tilting == "mean" or tilting == "MEAN":
        tilting = np.mean(x)
    elif tilting == "personal mean":
        tilting = np.mean(x, axis=0)

    if normalization == "STD":
        normalization = np.mean(x)
    elif normalization == "personal STD":
        normalization = np.mean(x, axis=0)
    return (x-tilting) / normalization


def plot_acc_lss(history_log):

    import matplotlib.pyplot as plt
    # list all data in history
    print(history_log.history.keys())
    # summarize history for accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history_log.history['categorical_accuracy'])
    plt.plot(history_log.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid(b=True, which='Both', axis='both')
    # summarize history for loss
    plt.subplot(1, 2, 2)
    plt.plot(history_log.history['loss'])
    plt.plot(history_log.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.grid(b=True, which='Both', axis='both')

    plt.show()
    plt.savefig('training_plot.png')


def horizontal_flip_and_show(data, labels, verbose=1):
    """
    tf.image.flip_left_right wrapper. adds left right augmented images and labels
    :param data: image tensor
    :param labels: one hot label tensor
    :param verbose: 1 to show random example
    :return: data with flipped addition, new labels
    """
    import tensorflow as tf
    fliped_data = tf.image.flip_left_right(data)
    if verbose == 1:
        import random
        import matplotlib.pyplot as plt
        rand = random.randint(0, 1000)
        plt.subplot(1, 2, 1)
        plt.title('Original Image #{}'.format(rand))
        plt.imshow(data[rand, :, :, :])
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.title('Augmented Image #{}'.format(rand))
        plt.imshow(fliped_data[rand, :, :, :])
        plt.axis('off')
        plt.show()
    return tf.concat([data, fliped_data], 0), tf.concat([labels, labels], 0)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix'):
    """
    from deeplizard
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    if normalize:
        cm = cm.astype('float32') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.imshow(cm, interpolation='nearest', cmap="Blues_r", aspect='auto')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation='vertical')
    plt.yticks(tick_marks, classes)

    thresh = 0.2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j], 2), horizontalalignment="center", verticalalignment="center",
                 color="black" if cm[i, j] > thresh else "white")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    plt.savefig('confusion_matrix_plot.png')
