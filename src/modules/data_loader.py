import pickle

import numpy as np


def load_dataset(data_path, split_factor=0.7):
    """Loads the dataset using a pickle file.

    Args:
        data_path (string): path to the pickle file.
        split_factor (float, optional): Factor to split train into train and val if required. Defaults to 0.7.

    Returns:
        trainX (array): Array that contains the training data.
        trainY (array): Array that contains the training labels.
        valX (array): Array that contains the validation data.
        valY (array): Array that contains the validation labels.
        testX (array): Array that contains the test data.
        testY (array): Array that contains the test labels.
    """
    print('Loading data from file: %s' % (data_path))
    with open(data_path, 'rb') as pickleFile:
        d = pickle.load(pickleFile)
    if len(d) == 6:
        # validation exists
        trainX, trainY, valX, valY, testX, testY = d[0], d[1], d[2], d[3], d[4], d[5]
    elif len(d) == 4:
        # no validation set
        trainX, trainY, testX, testY = d[0], d[1], d[2], d[3]
        valX, valY = None, None
    else:
        print("No valid dataset")
        exit()
    # convert the labels to categorial if they are one-hot encoded
    if len(trainY.shape) > 1:
        trainY = np.argmax(trainY, axis=1)
        testY = np.argmax(testY, axis=1)
        if not valY is None:
            valY = np.argmax(valY, axis=1)
    else:
        trainY = np.asarray(trainY, dtype=int)
        testY = np.asarray(testY, dtype=int)
        if not valY is None:
            valY = np.asarray(valY, dtype=int)
    # create vlaidation set if not existing
    if valY is None:
        trainX, trainY, valX, valY = split_dataset(
            trainX, trainY, split_factor)

    return trainX, trainY, valX, valY, testX, testY


def split_dataset(set_data, set_labels, split_factor):
    """Splits the given set into two sets using the defined split factor.

    Args:
        set_data (array): Array containing the data.
        set_labels (array): array containing the labels.
        split_factor (float): factor to split the data.

    Returns:
        set_1X (array): Array that contains the first part of the data.
        set_1Y (array): Array that contains the first part of the labels.
        set_2X (array): Array that contains the second part of the data.
        set_2Y (array): Array that contains the second part of the labels.
    """
    print("Splitting the data with factor:", str(split_factor))
    set_split = int(np.ceil(len(set_data) * split_factor))

    set_1X = set_data[:set_split, :]
    set_1Y = set_labels[:set_split]

    set_2X = set_data[set_split:, :]
    set_2Y = set_labels[set_split:]

    print('Split set into %s and %s elements' % (len(set_1X), len(set_2X)))
    return set_1X, set_1Y, set_2X, set_2Y


def convert_cannel_first(set_list):
    """Converts the data to channel first setup instead of channel last setup.

    Args:
        set_list (array): dataset with channel last setup.

    Returns:
        array: Data with channel first setup.
    """
    channels_first = []
    for dset in set_list:
        smax = len(dset.shape) - 1
        channels_first.append(np.transpose(
            dset, np.concatenate([[0], [smax], np.arange(1, smax)])))
    return channels_first


def shuffle_dataset(data, labels):
    """Shuffles the dataset.

    Args:
        data (array): Array with the data.
        labels (array): array with the labels.

    Returns:
        perm_data (array): Shuffled array with the data.
        perm_labels (array): Shuffled array with the labels.
    """
    perm = np.random.permutation(np.arange(data.shape[0]))
    perm_data = data[perm]
    perm_labels = labels[perm]
    return perm_data, perm_labels


def concatenate_channels(data):
    """Concatenates the channels of a dataset

    Args:
        data (array): Array with the data and shape length 3.

    Returns:
        array: Data with shape length 2.
    """
    data = np.reshape(data, (data.shape[0], data.shape[1] * data.shape[2]))
    return data


def prepare_dataset(data_path, shuffle=True, seed=0, verbose=1):
    """Prepares the dataset for the P2ExNet pipeline. This includes concatenating the channels and shuffling.

    Args:
        data_path (string): Data path to load the picklle file.
        shuffle (bool, optional): Boolean to shuffle the data. Defaults to True.
        seed (int, optional): Random seed for the shuffling. Defaults to 0.
        verbose (int, optional): Flag to show the outputs. Defaults to 1.

    Returns:
        dataset_dict (dict): Dataset dicht with data and parameters
    """
    train_x, train_y, val_x, val_y, test_x, test_y = load_dataset(data_path)

    if verbose:
        print('Train Shape:\t', train_x.shape)
        print('Val Shape:\t', val_x.shape)
        print('Test Shape:\t', test_x.shape)

    # data reshape
    input_width, n_input_channel = train_x.shape[1:]
    # concatenate the channels
    train_x = concatenate_channels(train_x)
    val_x = concatenate_channels(val_x)
    test_x = concatenate_channels(test_x)

    # one hot encoding
    n_classes = len(np.unique(train_y))
    train_y = np.eye(n_classes)[train_y]
    val_y = np.eye(n_classes)[val_y]
    test_y = np.eye(n_classes)[test_y]

    if verbose:
        print('Number of classes:\t', n_classes)

    # shuffle
    if shuffle:
        np.random.seed(seed)
        train_x, train_y = shuffle_dataset(train_x, train_y)
        val_x, val_y = shuffle_dataset(val_x, val_y)
        test_x, test_y = shuffle_dataset(test_x, test_y)

    # update global dictionary
    dataset_dict = {}
    dataset_dict['train_x'] = train_x
    dataset_dict['train_y'] = train_y
    dataset_dict['val_x'] = val_x
    dataset_dict['val_y'] = val_y
    dataset_dict['test_x'] = test_x
    dataset_dict['test_y'] = test_y
    dataset_dict['n_classes'] = n_classes
    dataset_dict['input_width'] = input_width
    dataset_dict['n_input_channel'] = n_input_channel

    return dataset_dict
