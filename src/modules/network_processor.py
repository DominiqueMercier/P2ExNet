import math
import os
import sys
import time

import numpy as np
import tensorflow as tf

from modules.autoencoder_helpers import print_and_write
from modules.plot_manager import (plot_build_org_comparison,
                                  plot_class_distribution,
                                  plot_class_distribution_full,
                                  visualize_prototype_images,
                                  visualize_prototypes)


def get_receptive_parameter(n_input_steps, widths, strides, paddings):
    """Computes the receptive parameters to match batch the encoded data to the input space.

    Args:
        n_input_steps (int): Number of input steps.
        widths (list): List with the widths.
        strides (list): List containing the stride.
        paddings (list): List with the paddings.

    Returns:
        currentLayer (list): List that contains the parameters computed to understand the receptive field.

    """
    def outFromIn(conv, layerIn):
        n_in = layerIn[0]
        j_in = layerIn[1]
        r_in = layerIn[2]
        start_in = layerIn[3]
        k = conv[0]
        s = conv[1]
        p = conv[2]

        n_out = math.floor((n_in - k + 2*p)/s) + 1
        actualP = (n_out-1)*s - n_in + k
        pR = math.ceil(actualP/2)
        pL = math.floor(actualP/2)

        j_out = j_in * s
        r_out = r_in + (k - 1)*j_in
        start_out = start_in + ((k-1)/2 - pL)*j_in
        return n_out, j_out, r_out, start_out

    net = []
    for i in range(len(widths)):
        net.append([widths[i], strides[i], paddings[i]])
    currentLayer = [n_input_steps, 1, 1, 1]
    for i in range(len(net)):
        currentLayer = outFromIn(net[i], currentLayer)
    return currentLayer


def get_se(idx, start, jump_size, rf_size, patch, input_width):
    """Computes the start and end of a patch within the data.

    Args:
        idx (int): The id of the patch.
        start (int): The start point of the processing.
        jump_size (int): The number of steps to jump.
        rf_size (int): The size of the receptive field.
        patch (int): The length of a patch.
        input_width (int): The length of the series.

    Returns:
        list: List that contains the start and end of the patch.
    """
    s = int(np.max(
        [0, start + idx * jump_size - rf_size]))
    e = int(np.min([start +
                    (idx + patch-1) *
                    jump_size + rf_size, input_width]))
    return [s, e]


def init_model(sess, saver, options, options_ex, dataset_dict, network_dict, tensor_dict):
    """Initialize the modle and computes required receptive field parameters.

    Args:
        sess (session): The tensorflow session used to compute everything.
        saver (object): The saver that handles the hyperparamter and other variables.
        options (dict): Option parser parameters.
        options_ex (dict): advanxed options.
        dataset_dict (dict): Dataset and parameters.
        network_dict (dict): network parameters.
        tensor_dict (dict): tensor dictionary.
    """
    # we compute the number of batches because both training and evaluation
    # happens batch by batch; we do not throw the entire test set onto the GPU
    network_dict['n_train_batch'] = dataset_dict['train_x'].shape[0] // options.batch_size
    network_dict['n_valid_batch'] = dataset_dict['val_x'].shape[0] // options.batch_size
    network_dict['n_test_batch'] = dataset_dict['test_x'].shape[0] // options.batch_size

    # predefine
    if options.predefine and not options.no_int:
        sess.run(tensor_dict['predefine'])

    # restore model
    if not options.train:
        saver.restore(sess, tf.train.latest_checkpoint(
            os.path.join(options_ex['checkpoint_folder'])))
        print('Model loaded')

    # frozen optimizer
    if options.no_freeze or options.no_int:
        network_dict['current_optimizer'] = 'optimizer'
    else:
        network_dict['current_optimizer'] = 'optimizer_proto'

    paddings = [(i - 1) / 2 for i in network_dict['f_x']]

    latent_paras = get_receptive_parameter(
        dataset_dict['input_width'], network_dict['f_x'], network_dict['stride_x'], paddings)
    jump_size = latent_paras[1]
    receptive_field = latent_paras[2]
    rf_size = (latent_paras[2] - 1) / 2
    start = latent_paras[3]

    # update dictionary
    network_dict['jump_size'] = jump_size
    network_dict['rf_size'] = rf_size
    network_dict['start'] = start


def process_blackbox_epoch(sess, console_log, options, options_ex, dataset_dict, network_dict, tensor_dict, opti=None, name='training'):
    """Processes all patches for the blackbox model for training or inference.

    Args:
        sess (session): The tensorflow session used to compute everything.
        console_log (file): File to log the console output.
        options (dict): Option parser parameters.
        options_ex (dict): additional parameters.
        dataset_dict (dict): Dataset and parameters.
        network_dict (dict): network dictionary.
        tensor_dict (dict): defined tensors.
        opti (object): optimizer.
        name (str, optional): Label for the console log. Should be the dataset name e.g training, validation or test. Defaults to 'training'.
    """
    if name == 'training':
        n_batch = network_dict['n_train_batch']
        x = dataset_dict['train_x']
        y = dataset_dict['train_y']
    elif name == 'validation':
        n_batch = network_dict['n_valid_batch']
        x = dataset_dict['val_x']
        y = dataset_dict['val_y']
    else:
        n_batch = network_dict['n_test_batch']
        x = dataset_dict['test_x']
        y = dataset_dict['test_y']

    start_time = time.time()
    ce, ac = 0.0, 0.0
    # Loop over all batches
    for i in range(n_batch):
        batch_x, batch_y = x[i*options.batch_size:(i+1) *
                             options.batch_size], y[i*options.batch_size:(i+1)*options.batch_size]
        # training
        if not opti is None:
            _, ce, ac = sess.run((opti, tensor_dict['class_error'], tensor_dict['accuracy']),
                                 feed_dict={tensor_dict['X']: batch_x, tensor_dict['Y']: batch_y, tensor_dict['lambda_class_t']: options_ex['lambda_class']})
        else:
            # inference
            ce, ac = sess.run((tensor_dict['class_error'], tensor_dict['accuracy']),
                              feed_dict={tensor_dict['X']: batch_x, tensor_dict['Y']: batch_y, tensor_dict['lambda_class_t']: options_ex['lambda_class']})
        ce += (ce/n_batch)
        ac += (ac/n_batch)
    end_time = time.time()
    print_and_write(
        name + ' takes {0:.2f} seconds.'.format((end_time - start_time)), console_log)
    # after every epoch, check the error terms on the entire training set
    print_and_write(name + " set errors:", console_log)
    print_and_write("\tclassification error:\t{:.6f}".format(ce), console_log)
    print_and_write("\taccuracy:\t\t{:.4f}".format(ac), console_log)


def process_blackbox(sess, saver, console_log, options, options_ex, dataset_dict, network_dict, tensor_dict):
    """Trains the black box model.

    Args:
        sess (session): The tensorflow session used to compute everything.
        saver (object): The saver that handles the hyperparamter and other variables.
        console_log (file): File to log the console output.
        options (dict): Option parser parameters.
        options_ex (dict): additional parameters.
        dataset_dict (dict): Dataset and parameters.
        network_dict (dict): network dictionary.
        tensor_dict (dict): defined tensors.
    """
    test_display_step = options.epochs // 10
    save_step = options.epochs // 10

    # Training cycle
    for epoch in range(options.epochs):

        print_and_write("#"*80, console_log)
        print_and_write("Epoch: %04d" % (epoch), console_log)

        # training
        process_blackbox_epoch(sess, console_log, options, options_ex, dataset_dict, network_dict, tensor_dict,
                               opti=tensor_dict['optimizer'], name='training')

        # validation
        process_blackbox_epoch(sess, console_log, options, options_ex, dataset_dict, network_dict, tensor_dict,
                               opti=None, name='validation')

        # test set accuracy evaluation
        if epoch % test_display_step == 0 or epoch == options.epochs - 1:
            process_blackbox_epoch(sess, console_log, options, options_ex, dataset_dict, network_dict, tensor_dict,
                                   opti=None, name='test')

        # save the model
        if epoch % save_step == 0 or epoch == options.epochs - 1:
            # one .meta file is enough to recover the computational graph
            saver.save(sess, os.path.join(options_ex['checkpoint_folder'], options_ex['model_filename']),
                       global_step=epoch,
                       write_meta_graph=(epoch == 0 or epoch == options.epochs - 1))

    print_and_write("Optimization Finished!", console_log)
    console_log.close()


def proces_interpretable_epoch(sess, console_log, options, options_ex, dataset_dict, network_dict, tensor_dict, opti=None, name='training'):
    """Processes all patches for the interpretable model for training or inference.

    Args:
        sess (session): The tensorflow session used to compute everything.
        console_log (file): File to log the console output.
        options (dict): Option parser parameters.
        options_ex (dict): additional parameters.
        dataset_dict (dict): Dataset and parameters.
        network_dict (dict): network dictionary.
        tensor_dict (dict): defined tensors.
        opti (object): optimizer.
        name (str, optional): Label for the console log. Should be the dataset name e.g training, validation or test. Defaults to 'training'.
    """
    if name == 'training':
        n_batch = network_dict['n_train_batch']
        x = dataset_dict['train_x']
        y = dataset_dict['train_y']
    elif name == 'validation':
        n_batch = network_dict['n_valid_batch']
        x = dataset_dict['val_x']
        y = dataset_dict['val_y']
    else:
        n_batch = network_dict['n_test_batch']
        x = dataset_dict['test_x']
        y = dataset_dict['test_y']

    start_time = time.time()
    ce, ae, p2s, s2p, div, clst, sep, te, ac = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # Loop over all batches
    for i in range(n_batch):
        batch_x, batch_y = x[i*options.batch_size:(
            i+1)*options.batch_size], y[i*options.batch_size:(i+1)*options.batch_size]
        if not opti is None:
            _, ce, ae, p2s, s2p, div, clst, sep, te, ac = \
                sess.run((opti, tensor_dict['class_error'], tensor_dict['ae_error'], tensor_dict['proto2sample_error'], tensor_dict['sample2proto_error'],
                          tensor_dict['diversity_error'], tensor_dict['clst_error'], tensor_dict['sep_error'], tensor_dict['total_error'], tensor_dict['accuracy']),
                         feed_dict={tensor_dict['X']: batch_x, tensor_dict['Y']: batch_y, tensor_dict['lambda_class_t']: options_ex['lambda_class'],
                                    tensor_dict['lambda_ae_t']: options_ex['lambda_ae'], tensor_dict['lambda_p2s_t']: options_ex['lambda_p2s'],
                                    tensor_dict['lambda_s2p_t']: options_ex['lambda_s2p'], tensor_dict['lambda_div_t']: options_ex['lambda_div'],
                                    tensor_dict['lambda_clst_t']: options_ex['lambda_clst'], tensor_dict['lambda_sep_t']: options_ex['lambda_sep']})
        else:
            ce, ae, p2s, s2p, div, clst, sep, te, ac = \
                sess.run((tensor_dict['class_error'], tensor_dict['ae_error'], tensor_dict['proto2sample_error'], tensor_dict['sample2proto_error'],
                          tensor_dict['diversity_error'], tensor_dict['clst_error'], tensor_dict['sep_error'], tensor_dict['total_error'], tensor_dict['accuracy']),
                         feed_dict={tensor_dict['X']: batch_x, tensor_dict['Y']: batch_y, tensor_dict['lambda_class_t']: options_ex['lambda_class'],
                                    tensor_dict['lambda_ae_t']: options_ex['lambda_ae'], tensor_dict['lambda_p2s_t']: options_ex['lambda_p2s'],
                                    tensor_dict['lambda_s2p_t']: options_ex['lambda_s2p'], tensor_dict['lambda_div_t']: options_ex['lambda_div'],
                                    tensor_dict['lambda_clst_t']: options_ex['lambda_clst'], tensor_dict['lambda_sep_t']: options_ex['lambda_sep']})
        ce += (ce/n_batch)
        ae += (ae/n_batch)
        p2s += (p2s/n_batch)
        s2p += (s2p/n_batch)
        div += (div/n_batch)
        clst += (clst/n_batch)
        sep += (sep/n_batch)
        te += (te/n_batch)
        ac += (ac/n_batch)
    end_time = time.time()
    print_and_write(
        name + ' takes {0:.2f} seconds.'.format((end_time - start_time)), console_log)
    # after every epoch, check the error terms on the entire training set
    print_and_write(name + " set errors:", console_log)
    print_and_write("\tclassification error:\t{:.6f}".format(ce), console_log)
    print_and_write("\tautoencoder error:\t{:.6f}".format(ae), console_log)
    print_and_write("\tproto2sample error:\t{:.6f}".format(p2s), console_log)
    print_and_write("\tsample2proto error:\t{:.6f}".format(s2p), console_log)
    print_and_write("\tdiversity error:\t{:.6f}".format(div), console_log)
    print_and_write("\tclst error:\t\t{:.6f}".format(clst), console_log)
    print_and_write("\tsep error:\t\t{:.6f}".format(sep), console_log)
    print_and_write("\ttotal error:\t\t{:.6f}".format(te), console_log)
    print_and_write("\taccuracy:\t\t{:.4f}".format(ac), console_log)


def compute_prototype_representatives(sess, options, dataset_dict, network_dict, tensor_dict):
    """Compute the prototype images based on the training samples to find the best representative.

    Args:
        sess (session): The tensorflow session used to compute everything.
        options (dict): Option parser parameters.
        dataset_dict (dict): Dataset and parameters.
        network_dict (dict): network dictionary.
        tensor_dict (dict): defined tensors.

    Returns:
        prototype_imgs (list): List of the prototype images based on the training samples.
        best_prototype_distance (list). List of the best distance between representative and real prototype.
    """
    prototype_imgs = [[] for i in range(options.n_prototypes)]
    best_sample_patch = np.zeros((options.n_prototypes, 2), dtype=int)
    best_prototype_distance = np.full(
        (options.n_prototypes), sys.float_info.max)
    for i in range(network_dict['n_train_batch']):
        batch_x, batch_y = dataset_dict['train_x'][i*options.batch_size:(
            i+1)*options.batch_size], dataset_dict['train_y'][i*options.batch_size:(i+1)*options.batch_size]
        distances, indices, samples = sess.run([tensor_dict['prototype_distances_reduce'], tensor_dict['prototype_distances_arg_reduce'], tensor_dict['X_img']],
                                               feed_dict={tensor_dict['X']: batch_x, tensor_dict['Y']: batch_y})
        # finding best patch over corpus
        for sidx, dist in enumerate(distances):
            for pidx, pDist in enumerate(dist):
                if best_prototype_distance[pidx] > pDist:
                    best_prototype_distance[pidx] = pDist
                    best_sample_patch[pidx, 0] = sidx
                    best_sample_patch[pidx,
                                      1] = indices[sidx, pidx]
                    s, e = get_se(
                        indices[sidx, pidx], network_dict['start'], network_dict['jump_size'], network_dict['rf_size'], options.patch_size, dataset_dict['input_width'])
                    prototype_imgs[pidx] = samples[sidx, s:e, :]
    return prototype_imgs, best_prototype_distance


def process_interpretable(sess, saver, console_log, options, options_ex, dataset_dict, network_dict, tensor_dict):
    """Trains the interpretable model.

    Args:
        sess (session): The tensorflow session used to compute everything.
        saver (object): The saver that handles the hyperparamter and other variables.
        console_log (file): File to log the console output.
        options (dict): Option parser parameters.
        options_ex (dict): additional parameters.
        dataset_dict (dict): Dataset and parameters.
        network_dict (dict): network dictionary.
        tensor_dict (dict): defined tensors.
    """
    test_display_step = options.epochs // 10
    save_step = options.epochs // 10
    opti = tensor_dict[network_dict['current_optimizer']]

    # Training cycle
    for epoch in range(options.epochs):
        # unfrezze
        if epoch > options.epochs * 0.8 and not options.no_freeze:
            opti = tensor_dict['optimizer_convex']

        print_and_write("#"*80, console_log)
        print_and_write("Epoch: %04d" % (epoch), console_log)

        # training
        proces_interpretable_epoch(sess, console_log, options, options_ex, dataset_dict, network_dict, tensor_dict,
                                   opti=opti, name='training')

        # validation set error terms evaluation
        proces_interpretable_epoch(sess, console_log, options, options_ex, dataset_dict, network_dict, tensor_dict,
                                   opti=None, name='validation')

        # test set accuracy evaluation
        if epoch % test_display_step == 0 or epoch == options.epochs - 1:
            proces_interpretable_epoch(sess, console_log, options, options_ex, dataset_dict, network_dict, tensor_dict,
                                       opti=None, name='test')

        # save the model
        if epoch % save_step == 0 or epoch == options.epochs - 1:
            # one .meta file is enough to recover the computational graph
            saver.save(sess, os.path.join(options_ex['checkpoint_folder'], options_ex['model_filename']),
                       global_step=epoch,
                       write_meta_graph=(epoch == 0 or epoch == options.epochs - 1))

            # compute representative prototype images
            prototype_imgs, best_prototype_distance = compute_prototype_representatives(
                sess, options, dataset_dict, network_dict, tensor_dict)

            # visualize the prototype images
            visualize_prototypes(epoch, options.n_prototypes,
                                 prototype_imgs, options_ex['img_folder'])

            np.save(os.path.join(
                options_ex['model_folder'], 'prototypes.npy'), prototype_imgs)
            np.save(os.path.join(options_ex['model_folder'], 'prototype_distances.npy'),
                    best_prototype_distance)

            # weight matrix
            w = np.transpose(np.reshape(tensor_dict['last_layer']['w'].eval(session=sess,
                                                                            ), (network_dict['n_patches'], options.n_prototypes, dataset_dict['n_classes'])), [2, 1, 0])
            np.save(os.path.join(
                options_ex['model_folder'], 'weight-matrix.npy'), w)

            if not options.no_ae:
                # Applying encoding and decoding over a small subset of the training set
                examples_to_show = 10
                encode_decode = sess.run(tensor_dict['X_decoded'],
                                         feed_dict={tensor_dict['X']: dataset_dict['train_x'][:examples_to_show]})

                # Compare original images with their reconstructions
                plot_build_org_comparison(
                    dataset_dict, epoch, examples_to_show, encode_decode, options_ex['img_folder'])

    print_and_write("Optimization Finished!", console_log)
    console_log.close()


def decode_patch_prototypes(sess, options, dataset_dict, network_dict, tensor_dict):
    """The prototypes get calculated and then decoded.

    Args:
        sess (session): The tensorflow session used to compute everything.
        options (dict): Option parser parameters.
        dataset_dict (dict): Dataset and parameters.
        network_dict (dict): network dictionary.
        tensor_dict (dict): defined tensors.

    Returns:
        array: Array with the decoded prototypes.
    """
    # closest samples
    encoded_patches = [[] for i in range(options.n_prototypes)]
    patch_ids = np.zeros(options.n_prototypes, dtype=int)
    best_prototype_distance = np.full(
        (options.n_prototypes), sys.float_info.max)
    for i in range(network_dict['n_train_batch']):
        batch_x, batch_y = dataset_dict['train_x'][i*options.batch_size:(
            i+1)*options.batch_size], dataset_dict['train_y'][i*options.batch_size:(i+1)*options.batch_size]
        distances, indices, samples = sess.run([tensor_dict['prototype_distances_reduce'], tensor_dict['prototype_distances_arg_reduce'], network_dict['el_x'][-1]],
                                               feed_dict={tensor_dict['X']: batch_x, tensor_dict['Y']: batch_y})
        # finding best patch over corpus
        for sidx, dist in enumerate(distances):
            for pidx, pDist in enumerate(dist):
                if best_prototype_distance[pidx] > pDist:
                    best_prototype_distance[pidx] = pDist
                    patch_ids[pidx] = indices[sidx, pidx]
                    encoded_patches[pidx] = samples[sidx]
    encoded_patches = np.asarray(encoded_patches)

    # replace latent part
    latents = sess.run(tensor_dict['prototype_feature_vectors'])
    latents = np.reshape(latents, (-1, options.patch_size,
                                   network_dict['n_map_x'][-1]))
    for i in range(latents.shape[0]):
        for l in range(latents.shape[1]):
            encoded_patches[i, patch_ids[i]+l] = latents[i, l]

    flatten_size = network_dict['lx_shape'][-1][1] * \
        network_dict['lx_shape'][-1][2]
    enc_d = np.reshape(encoded_patches, (-1, flatten_size))
    dec = sess.run(
        tensor_dict['X_decoded'], feed_dict={tensor_dict['feature_vectors']: enc_d})
    dec_d = np.reshape(
        dec, (-1, dataset_dict['input_width'], dataset_dict['n_input_channel']))

    decoded_patches = []
    for i in range(options.n_prototypes):
        s, e = get_se(patch_ids[i], network_dict['start'],
                      network_dict['jump_size'], network_dict['rf_size'], options.patch_size, dataset_dict['input_width'])
        decoded_patches.append(dec_d[i, s:e])

    return np.asarray(decoded_patches)


def compute_prototypes_close(decoded_prototypes, options, dataset_dict, network_dict):
    """Computes the closest prototypes after flattening the decoded ones.

    Args:
        decoded_prototypes (array): Decoded prototypes.
        options (dict): Option parser parameters.
        dataset_dict (dict): Dataset and parameters.
        network_dict (dict): network dictionary.

    Returns:
        array: Arry that contains the decoded close prototypes.
    """

    decoded_prototypes_flat = []
    for p in decoded_prototypes:
        decoded_prototypes_flat.append(p.flatten())

    decoded_distances = np.repeat(sys.float_info.max, options.n_prototypes)
    decoded_prototypes_close = [[] for _ in range(options.n_prototypes)]
    for series in dataset_dict['train_x']:
        series_r = np.reshape(
            series, (dataset_dict['input_width'], dataset_dict['n_input_channel']))
        for pid in range(options.n_prototypes):
            p = decoded_prototypes[pid]
            p_f = decoded_prototypes_flat[pid]
            for idx in range(network_dict['n_patches']):
                s, e = get_se(idx, network_dict['start'], network_dict['jump_size'],
                              network_dict['rf_size'], options.patch_size, dataset_dict['input_width'])
                part = series_r[s:e]
                if len(part) < len(p):
                    ex = len(p) - len(part)
                    part = np.concatenate(
                        [part, np.zeros((ex, dataset_dict['n_input_channel']))])
                elif len(part) > len(p):
                    part = part[:len(p)]
                part_f = part.flatten()
                mse = (np.square(p_f - part_f)).mean()
                if mse < decoded_distances[pid]:
                    decoded_distances[pid] = mse
                    decoded_prototypes_close[pid] = series_r[s:e]
    return decoded_prototypes_close


def compute_best_representative(sess, options, dataset_dict, network_dict, tensor_dict):
    """Computes the best representative prototype.

    Args:
        sess (seesion): Tensorflow session.
        options (dict): Option parser parameters.
        dataset_dict (dict): Dataset and parameters.
        network_dict (dict): network dictionary.
        tensor_dict (dict): defined tensors.

    Returns:
        best_prototype_distance (array): Array with the best distances.
        best_prototype_patch (array): Array with the best prototypes.
        prot (array): Prototype feature vector.

    """
    best_prototype_distance = np.full(
        (options.n_prototypes), sys.float_info.max)
    best_prototype_patch = np.zeros(
        (options.n_prototypes, network_dict['n_features']))
    for i in range(network_dict['n_train_batch']):
        batch_x, batch_y = dataset_dict['train_x'][i*options.batch_size:(
            i+1)*options.batch_size], dataset_dict['train_y'][i*options.batch_size:(i+1)*options.batch_size]
        distances, indices, samples, lat, prot = sess.run([tensor_dict['prototype_distances_reduce'], tensor_dict['prototype_distances_arg_reduce'],
                                                           tensor_dict['X_img'], tensor_dict['feature_patches_flat'], tensor_dict['prototype_feature_vectors']],
                                                          feed_dict={tensor_dict['X']: batch_x, tensor_dict['Y']: batch_y})
        # finding best patch over corpus
        for sidx, dist in enumerate(distances):
            for pidx, pDist in enumerate(dist):
                if best_prototype_distance[pidx] > pDist:
                    best_prototype_distance[pidx] = pDist
                    best_prototype_patch[pidx] = lat[sidx, indices[sidx, pidx]]
    return best_prototype_distance, best_prototype_patch, prot


def load_true_prototype_imgs(options, model_folder):
    """Loads the true prototypes form the correct file.

    Args:
        options (dict): option dictionary.
        model_folder (string): model folder.

    Returns:
        arrray: values of the true prototype images
    """
    if options.decode_latent:
        true_prototype_imgs = np.load(
            os.path.join(model_folder, 'prototypes_decoded.npy'), allow_pickle=True)
    elif options.decode_latent_close:
        true_prototype_imgs = np.load(
            os.path.join(model_folder, 'prototypes_decoded_close.npy'), allow_pickle=True)
    else:
        true_prototype_imgs = np.load(
            os.path.join(model_folder, 'prototypes.npy'), allow_pickle=True)
    return true_prototype_imgs


def get_patch(x, n_prototypes):
    return x // n_prototypes


def get_proto(x, n_prototypes):
    return x % n_prototypes


def create_sidx_object(options, network_dict, n_classes, sidx, nb, batch_data):
    """Creates the sidx dictionary with all data for the evaluation and building.

    Args:
        options (dict): dictionary with the options.
        network_dict (dict): network parameters.
        n_classes (int): number of classes
        sidx (int): number of the sample.
        nb (int): number of the batch.
        batch_data (dict): dict with the computed data for the batch.

    Returns:
        dict: sidx data dictionary.
    """
    batch_y, distances, indices, distances_full, samples, pred, w = batch_data

    sidx_dict = {}
    sidx_dict['sidx_num'] = nb*options.batch_size + sidx
    sidx_dict['sidx_dist'] = distances[sidx] if not options.full_build else distances_full[sidx]
    sidx_dict['sidx_ind'] = indices[sidx]
    sidx_dict['sidx_pred'] = np.argmax(pred[sidx])
    sidx_dict['sidx_x'] = samples[sidx]
    sidx_dict['sidx_gt'] = np.argmax(batch_y[sidx])

    # get weights
    if not options.full_build:
        w_squeeze = np.reshape(
            w, [network_dict['n_patches'], options.n_prototypes, n_classes])
        w_squeeze = np.asarray(
            [w_squeeze[sidx_dict['sidx_ind'][i]][i] for i in range(options.n_prototypes)])

    else:
        w_squeeze = np.reshape(
            w, [network_dict['n_patches'] * options.n_prototypes, n_classes])

        # first patches then prototypes
        sidx_dict['sidx_dist'] = np.reshape(sidx_dict['sidx_dist'], (-1))

    sidx_dict['w_squeeze'] = w_squeeze

    sidx_dict['w_squeeze_pred'] = w_squeeze[:, sidx_dict['sidx_pred']]
    sidx_dict['sidx_w'] = sidx_dict['w_squeeze_pred'] * \
        np.reciprocal(sidx_dict['sidx_dist'])

    if options.sort_val:
        sidx_w_argsort = np.argsort(abs(sidx_dict['sidx_w']))[::-1]
    else:
        sidx_w_argsort = np.argsort(sidx_dict['sidx_dist'])

    sidx_dict['sidx_w_argsort'] = sidx_w_argsort

    return sidx_dict


def produce_build(options, dataset_dict, network_dict, sidx_dict, actual_n_prototypes, true_prototype_imgs):
    """Creates the actual build using the prototypes to replace the data by representatives.

    Args:
        options (dict): dictionary with the options.
        dataset_dict (dict): dataset parameters.
        network_dict (dict): network parameters.
        sidx_dict (dict): samples dictionary
        actual_n_prototypes (int): the true prototype number based on mode.

    Returns:
        array: buiilds using the prototypes to replace data
    """
    sidx_dict['sidx_build'] = np.zeros(sidx_dict['sidx_x'].shape)
    sidx_dict['sidx_build_dif'] = 0
    sidx_dict['sidx_build_set'] = set(np.arange(dataset_dict['input_width']))
    sidx_dict['sidx_build_se_list'] = []
    sidx_dict['sidx_build_replaced'] = set()

    prototype_adjusted_imgs = [[]
                               for i in range(actual_n_prototypes)]
    for idx in range(actual_n_prototypes):
        pidx = sidx_dict['sidx_w_argsort'][idx]
        s, e = get_se(
            sidx_dict['sidx_ind'][pidx] if not options.full_build else get_patch(
                pidx, options.n_prototypes), network_dict['start'], network_dict['jump_size'],
            network_dict['rf_size'], options.patch_size, dataset_dict['input_width'])
        tmp_img = true_prototype_imgs[pidx if not options.full_build else get_proto(
            pidx, options.n_prototypes)]
        # adjust prototype
        if options.same_scale:
            part = sidx_dict['sidx_x'][s:tmp_img.shape[0] + s, :]
            mi_s, ma_s, mi_p, ma_p = np.min(part), np.max(
                part), np.min(tmp_img), np.max(tmp_img)
            tmp_img = ((tmp_img - mi_p) /
                       (ma_p - mi_p)) * (ma_s - mi_s) + mi_s

        prototype_adjusted_imgs[pidx] = tmp_img

        # include into build
        if idx > np.ceil(options.n_prototypes // 5) and options.build_mode == 1:
            continue
        if sidx_dict['sidx_w'][pidx] < 0 and options.build_mode == 2:
            continue
        if sidx_dict['sidx_w'][pidx] > 0 and options.build_mode == 3:
            continue
        if idx > options.build_cut-1:
            continue

        # adjust end to prevent out ot index
        end = tmp_img.shape[0]
        if s + end > sidx_dict['sidx_build'].shape[0] - 1:
            end -= (tmp_img.shape[0] +
                    s) - sidx_dict['sidx_build'].shape[0]

        # fill if not filled yet to prevent overlaps
        range_set = set(np.arange(s, e))
        if range_set.issubset(sidx_dict['sidx_build_set']):
            sidx_dict['sidx_build'][s:s+end, :] = tmp_img[0:end, :]
            sidx_dict['sidx_build_se_list'].append([s, s+end-1])
            sidx_dict['sidx_build_replaced'].add(pidx)
            sidx_dict['sidx_build_dif'] += int(
                100 * sidx_dict['sidx_dist'][pidx if not options.full_build else get_proto(pidx, options.n_prototypes)])/100
            sidx_dict['sidx_build_set'] -= range_set
    return prototype_adjusted_imgs


def update_build_evaluation(sess, options, dataset_dict, tensor_dict, sidx_dict, eval_dict, sidx, nb, sum_samples):
    """Udates the evaluation statistics using the build sample.

    Args:
        sess (session): tensorflow session.
        options (dict): dictionary with the options.
        dataset_dict (dict): dataset parameters.
        tensor_dict (dict): tensors used in the network.
        sidx_dict (dict): sample build parameters.
        eval_dict (dict): evaluation dictionary.
        sidx (int): number of the sample.
        nb (int): number of the batch.
        sum_samples (int): consumed samples.
    """
    # adjust remaining build
    fill_list = list(sidx_dict['sidx_build_set'])
    eval_dict['same_val'] += len(fill_list)
    if options.fill:
        for fid in fill_list:
            sidx_dict['sidx_build'][fid] = sidx_dict['sidx_x'][fid]

    # compute new build sample
    build_x = [np.reshape(
        sidx_dict['sidx_build'], sidx_dict['sidx_build'].shape[0] * sidx_dict['sidx_build'].shape[1])]
    build_pred = sess.run(tensor_dict['logits'], feed_dict={
                          tensor_dict['X']: build_x})
    sidx_dict['build_pred'] = np.argmax(build_pred)

    # build statistics
    eval_dict['same_pred'] += 1 if sidx_dict['sidx_pred'] == sidx_dict['build_pred'] else 0
    eval_dict['corr_pred_default'] += 1 if sidx_dict['sidx_gt'] == sidx_dict['sidx_pred'] else 0
    eval_dict['corr_pred_build'] += 1 if sidx_dict['sidx_gt'] == sidx_dict['build_pred'] else 0
    eval_dict['diffs'] += sidx_dict['sidx_build_dif']

    pred_sum = (sidx+1) + (options.batch_size * nb)

    if pred_sum % (sum_samples // 20) == 0 or pred_sum == sum_samples or not options.no_plot:
        acc_same = int(
            (eval_dict['same_pred'] / pred_sum) * 10000) / 100
        acc_default = int(
            (eval_dict['corr_pred_default'] / pred_sum) * 10000) / 100
        acc_build = int(
            (eval_dict['corr_pred_build'] / pred_sum) * 10000) / 100

        val_sum = dataset_dict['input_width'] * pred_sum
        rep = int((1 - eval_dict['same_val'] / val_sum) * 10000) / 100

        avg_diff = int((eval_dict['diffs'] / pred_sum) * 100) / 100

        end_str = '\n' if pred_sum == sum_samples or not options.no_plot else '\r'

        print('Same:\t%s / %s | Replace:\t%s%% | Equal:\t%s%% | AccD:\t%s%% | AccB:\t%s%% | ADiff:\t%s' % (
            eval_dict['same_pred'], pred_sum, rep, acc_same, acc_default, acc_build, avg_diff), end=end_str)


def evaluate_and_build(sess, options, options_ex, dataset_dict, network_dict, tensor_dict):
    """Evaluates and builld the prototypes.

    Args:
        sess (seesion): Tensorflow session.
        options (dict): Option parser parameters.
        options_ex (dict): additional parameters.
        dataset_dict (dict): datset parameters.
        network_dict (dict): network dictionary.
        tensor_dict (dict): tensors used by the model.

    """
    # mandatory metrics
    true_prototype_imgs = load_true_prototype_imgs(
        options, options_ex['model_folder'])

    # init variables for build

    eval_dict = {}
    eval_dict['same_pred'] = 0
    eval_dict['corr_pred_default'] = 0
    eval_dict['corr_pred_build'] = 0
    eval_dict['same_val'] = 0
    eval_dict['diffs'] = 0

    if options.build_compare:
        build_samples = []
    sum_samples = network_dict['n_test_batch'] * options.batch_size
    if options.full_build:
        actual_n_prototypes = network_dict['n_patches'] * \
            options.n_prototypes
    else:
        actual_n_prototypes = options.n_prototypes
    print('#'*80)
    print('Build Statistics')

    for nb in range(network_dict['n_test_batch']):
        batch_x, batch_y = dataset_dict['test_x'][nb*options.batch_size:(
            nb+1)*options.batch_size], dataset_dict['test_y'][nb*options.batch_size:(nb+1)*options.batch_size]
        distances, indices, distances_full, samples, pred, w = sess.run(
            [tensor_dict['prototype_distances_reduce'], tensor_dict['prototype_distances_arg_reduce'], tensor_dict['prototype_distances'],
                tensor_dict['X_img'], tensor_dict['logits'], tensor_dict['last_layer']['w']],
            feed_dict={tensor_dict['X']: batch_x, tensor_dict['Y']: batch_y})

        # evaluate for each sample
        for sidx in range(options.batch_size):
            # sidx objects
            batch_data = [batch_y, distances, indices,
                          distances_full, samples, pred, w]
            sidx_dict = create_sidx_object(
                options, network_dict, dataset_dict['n_classes'], sidx, nb, batch_data)

            if options.show_distribution:
                # overall
                plot_class_distribution(options, sidx_dict)
                # patchwise
                if options.full:
                    plot_class_distribution_full(
                        options, dataset_dict, network_dict, sidx_dict)

            prototype_adjusted_imgs = produce_build(
                options, dataset_dict, network_dict, sidx_dict, actual_n_prototypes, true_prototype_imgs)

            update_build_evaluation(
                sess, options, dataset_dict, tensor_dict, sidx_dict, eval_dict, sidx, nb, sum_samples)

            if options.build_compare:
                build_samples.append([sidx_dict['sidx_x'], sidx_dict['sidx_build'],
                                      sidx_dict['sidx_gt'], sidx_dict['sidx_pred'], sidx_dict['build_pred']])

            if not (options.no_plot or options.build_compare):
                visualize_prototype_images(
                    options, dataset_dict, network_dict, sidx_dict, prototype_adjusted_imgs)

        if options.build_compare:
            if not options.decode_latent:
                np.save(os.path.join(options_ex['model_folder'],
                                     'build_compare.npy'), build_samples)
            else:
                np.save(os.path.join(options_ex['model_folder'],
                                     'build_compare_decoded.npy'), build_samples)
