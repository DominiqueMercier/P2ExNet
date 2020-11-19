import sys

import numpy as np
import tensorflow as tf

from modules.autoencoder_helpers import (list_of_distances,
                                         list_of_distances_patches,
                                         list_of_norms)
from modules.model import deconv_layer


def create_decoder(options, dataset_dict, network_dict, tensor_dict):
    """Creates the decoder part of the network

    Args:
        options (dict): Option parser parameters.
        dataset_dict (dict): Dataset and parameters.
        network_dict (dict): network parameters.
        tensor_dict (dict): defined tensors.
    """

    '''deconv_batch_size is the number of feature vectors in the batch going into
    the deconvolutional network. This is required by the signature of
    conv2d_transpose. But instead of feeding in the value, the size is infered during
    sess.run by looking at how many rows the feature_vectors matrix has
    '''
    deconv_batch_size = tf.identity(tf.shape(tensor_dict['feature_vectors'])[
                                    0], name="deconv_batch_size")

    # this is necessary for prototype images evaluation
    reshape_feature_vectors = tf.reshape(
        tensor_dict['feature_vectors'], shape=[-1, network_dict['lx_shape'][-1][1], network_dict['lx_shape'][-1][2]])

    # dln means the output of the nth layer of the decoder
    dl_x = []
    for i in range(options.n_layers):
        i_rev = options.n_layers - (i+1)
        num = str(i_rev + 1)
        if options.n_layers == 1:
            dl = deconv_layer(reshape_feature_vectors, tensor_dict['weights']['dec_f' + num], tensor_dict['biases']['dec_b' + num],
                              output_shape=[deconv_batch_size,
                                            dataset_dict['input_width'], dataset_dict['n_input_channel']],
                              strides=network_dict['stride_x'][i_rev], padding="SAME", nonlinearity=tf.nn.sigmoid)
            dl_x.append(dl)
            continue
        if i == 0:
            dl = deconv_layer(reshape_feature_vectors, tensor_dict['weights']['dec_f' + num], tensor_dict['biases']['dec_b' + num],
                              output_shape=[deconv_batch_size,
                                            network_dict['lx_shape'][i_rev-1][1], network_dict['lx_shape'][i_rev-1][2]],
                              strides=network_dict['stride_x'][i_rev], padding="SAME")
            dl_x.append(dl)
        elif i < (options.n_layers - 1):
            dl = deconv_layer(dl_x[-1], tensor_dict['weights']['dec_f' + num], tensor_dict['biases']['dec_b' + num],
                              output_shape=[deconv_batch_size,
                                            network_dict['lx_shape'][i_rev-1][1], network_dict['lx_shape'][i_rev-1][2]],
                              strides=network_dict['stride_x'][i_rev], padding="SAME")
            dl_x.append(dl)
        else:
            dl = deconv_layer(dl_x[-1], tensor_dict['weights']['dec_f' + num], tensor_dict['biases']['dec_b' + num],
                              output_shape=[deconv_batch_size,
                                            dataset_dict['input_width'], dataset_dict['n_input_channel']],
                              strides=network_dict['stride_x'][i_rev], padding="SAME", nonlinearity=tf.nn.sigmoid)
            dl_x.append(dl)
    return dl_x


def assign_prototype_classes(n_prototypes, n_classes, n_patches, last_layer):
    """Method to assign the classes to the prototypes.

    Args:
        n_prototypes (int): Number of prototpes.
        n_classes (int): Number of classs.
        n_patches (int): Number of patches.
        last_layer (vector): The last layer of the network ( classification layer).

    Returns:
        cluster_labels (array): Labels of each prototype.
        per_lass (int): Number of prototypes per class.
    """
    # predefine weights to push prototypes to specific classes
    cluster_labels = np.zeros(n_prototypes, dtype=int)
    assign_data = np.full((n_prototypes, n_classes, n_patches), -0.5)
    per_class = int(np.ceil(n_prototypes / n_classes))
    for i in range(n_prototypes):
        ac = int(i / per_class)
        assign_data[i][ac] = np.ones(n_patches)
        cluster_labels[i] = ac
    predefine = tf.assign(last_layer['w'], np.reshape(np.transpose(
        assign_data, [2, 0, 1]), [n_patches*n_prototypes, n_classes]))
    return cluster_labels, per_class, predefine


def compute_distances(options, network_dict, tensor_dict, cluster_labels, per_class):
    """This method computes all distances used to train the network with the novel introduces loss that covers the different aspects.

    Args:
        options (dict): Option parser parameters.
        network_dict (dict): network parameters.
        tensor_dict (dict): defined tensors.
        cluster_labels (array): The labels of the preassigned prototypes.
        per_class (int): Number of prototypes per class.
    """

    '''
    prototype_distances is the list of distances from each x_i to every prototype
    in the latent space
    feature_vector_distances is the list of distances from each prototype to every x_i
    in the latent space
    '''
    prototype_distances = list_of_distances_patches(tensor_dict['feature_patches_flat'],
                                                    tensor_dict['prototype_feature_vectors'])

    prototype_distances = tf.identity(
        prototype_distances, name='prototype_distances')

    feature_vector_distances = list_of_distances_patches(tensor_dict['feature_patches_flat'],
                                                         tensor_dict['prototype_feature_vectors'], True)
    feature_vector_distances = tf.identity(
        feature_vector_distances, name='feature_vector_distances')

    prototype_diversities = list_of_distances(tensor_dict['prototype_feature_vectors'],
                                              tensor_dict['prototype_feature_vectors'])

    prototype_diversities = tf.matrix_set_diag(
        prototype_diversities, tf.fill([options.n_prototypes], sys.float_info.max))

    # reduce distances to best distance for sample
    prototype_distances_reduce = tf.reduce_min(prototype_distances, axis=1)
    prototype_distances_reduce = tf.identity(
        prototype_distances_reduce, name='prototype_distances_reduce')

    prototype_sim = tf.reciprocal(
        tf.reshape(prototype_distances, [-1, network_dict['n_patches']*options.n_prototypes]), name='prototype_sim')
    prototype_sim = tf.identity(
        prototype_sim, name='prototype_sim')

    feature_vector_distances_reduce = tf.reduce_min(
        feature_vector_distances, axis=2)
    feature_vector_distances_reduce = tf.identity(
        feature_vector_distances_reduce, name='prototype_distances_reduce')

    # reduce to get the argument for the best distance
    prototype_distances_arg_reduce = tf.argmin(prototype_distances, axis=1)
    prototype_distances_arg_reduce = tf.identity(
        prototype_distances_arg_reduce, name='prototype_distances_arg_reduce')

    feature_vector_distances_arg_reduce = tf.argmin(
        feature_vector_distances, axis=2)
    feature_vector_distances_arg_reduce = tf.identity(
        feature_vector_distances_arg_reduce, name='prototype_distances_arg_reduce')

    if options.predefine:
        # error 4 and 5
        def assign_dists(X, Y):
            """Computes the corresponding cluster and seperation loss.

            Args:
                X (vector): Input data.
                Y (vector): Label data.

            Returns:
                clst (array): Array with the clustering distances.
                sep (array): Array with the separation distances.
            """
            clst = np.zeros((options.batch_size, network_dict['n_patches'], per_class),
                            dtype=np.float32)
            sep = np.zeros((options.batch_size, network_dict['n_patches'], options.n_prototypes -
                            per_class), dtype=np.float32)
            for sid in range(options.batch_size):
                clst_idx = []
                sep_idx = []
                for cid, cl in enumerate(cluster_labels):
                    if cl == np.argmax(Y[sid]):
                        clst_idx.append(cid)
                    else:
                        sep_idx.append(cid)
                for pid in range(network_dict['n_patches']):
                    clst[sid, pid] = X[sid, pid][clst_idx]
                    sep[sid, pid] = X[sid, pid][sep_idx]
            return clst, sep

        clst_distances, sep_distances = tf.py_func(
            assign_dists, [prototype_distances, tensor_dict['Y']], [tf.float32, tf.float32])
        # reduce distances to best distance for sample
        clst_distances_reduce = tf.reduce_min(clst_distances, axis=1)
        clst_distances_reduce = tf.identity(
            clst_distances_reduce, name='clst_prototype_distances_reduce')
        sep_distances_reduce = tf.reduce_min(sep_distances, axis=1)
        sep_distances_reduce = tf.identity(
            sep_distances_reduce, name='sep_prototype_distances_reduce')

        # update tensor dict
        tensor_dict['clst_distances_reduce'] = clst_distances_reduce
        tensor_dict['sep_distances_reduce'] = sep_distances_reduce

    # update tensor dict
    tensor_dict['feature_vector_distances_reduce'] = feature_vector_distances_reduce
    tensor_dict['prototype_distances'] = prototype_distances
    tensor_dict['prototype_distances_reduce'] = prototype_distances_reduce
    tensor_dict['prototype_diversities'] = prototype_diversities
    tensor_dict['prototype_sim'] = prototype_sim
    tensor_dict['prototype_distances_arg_reduce'] = prototype_distances_arg_reduce


def define_composed_error(options, tensor_dict):
    """Computes the composed error including the lambdas to weight it.

    Args:
        options (dict): Option parser parameters.
        tensor_dict (dict): defined tensors.
    """
    '''
    the error function consists of 4 terms, the autoencoder loss,
    the classification loss, and the two requirements that every feature vector in
    X look like at least one of the prototype feature vectors and every prototype
    feature vector look like at least one of the feature vectors in X.
    '''

    if not options.no_ae:
        ae_error = tf.reduce_mean(list_of_norms(
            tensor_dict['X_decoded'] - tensor_dict['X_true']), name='ae_error')
    else:
        ae_error = tf.constant(0, name='ae_error')
    class_error = tf.losses.softmax_cross_entropy(
        onehot_labels=tensor_dict['Y'], logits=tensor_dict['logits'])
    class_error = tf.identity(class_error, name='class_error')
    proto2sample_error = tf.reduce_mean(tf.reduce_min(
        tensor_dict['feature_vector_distances_reduce'], axis=0), name='proto2sample_error')
    sample2proto_error = tf.reduce_mean(tf.reduce_min(
        tensor_dict['prototype_distances_reduce'], axis=1), name='sample2proto_error')
    diversity_error = tf.reciprocal(tf.math.log(
        1 + tf.reduce_mean(tf.reduce_min(tensor_dict['prototype_diversities'], axis=0), axis=0)), name='diversity_error')
    if options.predefine:
        clst_error = tf.reduce_mean(tf.reduce_min(
            tensor_dict['clst_distances_reduce'], axis=1), name='clst_error')
        sep_error = tf.reduce_mean(-tf.reduce_min(
            tensor_dict['sep_distances_reduce'], axis=1), name='sep_error')

    # total_error is the our minimization objective
    total_error = tensor_dict['lambda_class_t'] * class_error +\
        tensor_dict['lambda_p2s_t'] * proto2sample_error + \
        tensor_dict['lambda_s2p_t'] * sample2proto_error + \
        tensor_dict['lambda_div_t'] * diversity_error
    if not options.no_ae:
        total_error += tensor_dict['lambda_ae_t'] * ae_error
    if options.predefine:
        total_error += tensor_dict['lambda_clst_t'] * \
            clst_error + tensor_dict['lambda_sep_t'] * sep_error

    total_error = tf.identity(total_error, name='total_error')

    # update tensor dict
    tensor_dict['total_error'] = total_error
    tensor_dict['ae_error'] = ae_error
    tensor_dict['class_error'] = class_error
    tensor_dict['proto2sample_error'] = proto2sample_error
    tensor_dict['sample2proto_error'] = sample2proto_error
    tensor_dict['diversity_error'] = diversity_error
    tensor_dict['clst_error'] = clst_error
    tensor_dict['sep_error'] = sep_error


def create_accuracy(logits, Y):
    """Computes the accuracy

    Args:
        logits (vector): Vector with the logits.
        Y (vector): True labels.

    Returns:
        vector: Accuracy of the passed labels.
    """
    # accuracy is not the classification error term; it is the percentage accuracy
    correct_prediction = tf.equal(tf.argmax(logits, 1),
                                  tf.argmax(Y, 1),
                                  name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32),
                              name='accuracy')
    return accuracy


def create_interpretable_model(options, dataset_dict, network_dict, tensor_dict):
    """Creates the interpretable model. And returns the required measurements and optimizers.

    Args:
        options (dict): Option parser parameters.
        dataset_dict (dict): Dataset and parameters.
        network_dict (dict): network parameters.
        tensor_dict (dict): defined tensors.
    """
    input_size = dataset_dict['input_width'] * dataset_dict['n_input_channel']
    patch = options.patch_size
    n_features = patch * network_dict['lx_shape'][-1][2]
    flatten_size = network_dict['lx_shape'][-1][1] * \
        network_dict['lx_shape'][-1][2]
    network_dict['n_features'] = n_features

    network_dict['n_patches'] = network_dict['lx_shape'][-1][1]-patch+1
    feature_patches = tf.map_fn(lambda x: tf.reshape([tf.slice(x, [i, 0], [
                                patch, network_dict['lx_shape'][-1][2]]) for i in range(network_dict['n_patches'])],
        shape=[network_dict['n_patches'], patch, network_dict['lx_shape'][-1][2]]), network_dict['el_x'][-1])
    feature_patches_flat = tf.reshape(
        feature_patches, shape=[-1, network_dict['n_patches'], network_dict['n_features']], name='feature_patches')
    tensor_dict['feature_patches_flat'] = feature_patches_flat

    # feature vectors is the flattened output of the encoder
    feature_vectors = tf.reshape(
        network_dict['el_x'][-1], shape=[-1, flatten_size], name='feature_vectors')
    tensor_dict['feature_vectors'] = feature_vectors
    # the list prototype feature vectors
    prototype_feature_vectors = tf.Variable(tf.random.normal(shape=[options.n_prototypes, n_features],
                                                             dtype=tf.float32, stddev=0.01),
                                            name='prototype_feature_vectors')
    tensor_dict['prototype_feature_vectors'] = prototype_feature_vectors

    if not options.no_ae:
        dl_x = create_decoder(options, dataset_dict, network_dict, tensor_dict)

    # create classification layer
    last_layer = {
        'w': tf.Variable(tf.random_uniform(shape=[network_dict['n_patches']*options.n_prototypes, dataset_dict['n_classes']],
                                           dtype=tf.float32),
                         name='last_layer_w')
    }
    tensor_dict['last_layer'] = last_layer

    if options.predefine:
        cluster_labels, per_class, predefine = assign_prototype_classes(
            options.n_prototypes, dataset_dict['n_classes'], network_dict['n_patches'], tensor_dict['last_layer'])
        tensor_dict['predefine'] = predefine

    # include possible decoder output if decoder is used
    X_decoded, X_true = None, None
    if not options.no_ae:
        '''
        X_decoded is the decoding of the encoded feature vectors in X;
        we reshape it to match the shape of the training input
        X_true is the correct output for the autoencoder
        '''
        X_decoded = tf.reshape(
            dl_x[-1], shape=[-1, input_size], name='X_decoded')
        X_true = tf.identity(tensor_dict['X'], name='X_true')
    tensor_dict['X_decoded'] = X_decoded
    tensor_dict['X_true'] = X_true

    # compute all required distances
    compute_distances(options, network_dict, tensor_dict,
                      cluster_labels, per_class)

    # the logits are the weighted sum of distances from prototype_distances
    logits = tf.matmul(tensor_dict['prototype_sim'],
                       last_layer['w'], name='logits')
    tensor_dict['logits'] = logits
    probability_distribution = tf.nn.softmax(logits=logits,
                                             name='probability_distribution')

    define_composed_error(options, tensor_dict)

    accuracy = create_accuracy(logits, tensor_dict['Y'])

    # costum trainable variables
    optimizer = tf.train.AdamOptimizer(
        options.learning_rate).minimize(tensor_dict['total_error'])
    optimizer_convex = tf.train.AdamOptimizer(options.learning_rate).minimize(
        tensor_dict['total_error'], var_list=[last_layer['w']])
    optimizer_proto = tf.train.AdamOptimizer(options.learning_rate).minimize(
        tensor_dict['total_error'], var_list=[v for v in tf.trainable_variables() if v.name != 'last_layer_w:0'])
    # add the optimizer to collection so that we can retrieve the optimizer and resume training
    tf.add_to_collection("optimizer", optimizer)
    tf.add_to_collection("optimizer_convex", optimizer_convex)
    tf.add_to_collection("optimizer_proto", optimizer_proto)

    # update tensor_dict
    tensor_dict['accuracy'] = accuracy
    tensor_dict['optimizer'] = optimizer
    tensor_dict['optimizer_convex'] = optimizer_convex
    tensor_dict['optimizer_proto'] = optimizer_proto


def create_blackbox_model(options, dataset_dict, network_dict, tensor_dict):
    """Create the blackbox model including all evaluation metrics. This model has the same encoder structure 
    but replaces the interpretable part with a dense layer that has the same number of units as the interpretable has prototypes.

    Args:
        options (dict): Option parser parameters.
        dataset_dict (dict): Dataset and parameters.
        network_dict (dict): network parameters.
        tensor_dict (dict): defined tensors.
    """
    # add flatten layer after encoder part
    fl1 = tf.layers.flatten(network_dict['el_x'][-1])
    n_features = int(fl1.get_shape()[-1])
    # add denselayer
    fcl1 = tf.layers.dense(fl1, options.n_prototypes, 'relu')
    logits = tf.layers.dense(fcl1, dataset_dict['n_classes'], 'linear')
    # error
    class_error = tf.losses.softmax_cross_entropy(
        onehot_labels=tensor_dict['Y'], logits=logits)

    accuracy = create_accuracy(logits, tensor_dict['Y'])

    # costum trainable variables
    optimizer = tf.train.AdamOptimizer(
        options.learning_rate).minimize(class_error)
    tf.add_to_collection("optimizer", optimizer)

    # update tensor_dict
    tensor_dict['accuracy'] = accuracy
    tensor_dict['optimizer'] = optimizer
    tensor_dict['logits'] = logits
    tensor_dict['class_error'] = class_error

    network_dict['n_features'] = n_features
