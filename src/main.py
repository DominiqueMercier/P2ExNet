import os
from optparse import OptionParser

import numpy as np
import tensorflow as tf

from modules.autoencoder_helpers import makedirs
from modules.data_loader import prepare_dataset
from modules.model import create_model_architecture
from modules.model_generator import (create_blackbox_model,
                                     create_interpretable_model)
from modules.network_processor import (compute_best_representative,
                                       compute_prototypes_close,
                                       decode_patch_prototypes,
                                       evaluate_and_build, init_model,
                                       process_blackbox, process_interpretable)
from modules.plot_manager import plot_latent_prototypes, plot_prototype_images
from modules.statistic_processor import compute_prototype_distances

# set tensorflow session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def process(options):
    print("Start processing")

    # create required parameter dictionaries
    options_ex = {}
    network_dict = {}

    # dataset preparation
    dataset_dict = prepare_dataset(
        options.data_path, shuffle=options.shuffle, seed=options.seed)

    # setup directories and names
    options_ex['model_name'] = options.model_name + \
        "_l" + str(options.n_layers)
    if options.no_int:
        options_ex['model_name'] += "_nonInterpretable"
    else:
        options_ex['model_name'] += "_p" + str(options.patch_size)
        if options.no_ae:
            options_ex['model_name'] += "_noAE"
        else:
            options_ex['model_name'] += "_interpretable"

    # the directory to save the model
    options_ex['model_folder'] = os.path.join(
        os.getcwd(), "saved_model", options.data_name, options_ex['model_name'])
    makedirs(options_ex['model_folder'])

    options_ex['checkpoint_folder'] = os.path.join(
        options_ex['model_folder'], "checkpoints")
    makedirs(options_ex['checkpoint_folder'])

    options_ex['img_folder'] = os.path.join(options_ex['model_folder'], "img")
    makedirs(options_ex['img_folder'])

    options_ex['model_filename'] = 'network'

    # the maximum number of model snapshots we allow tensorflow to save to disk
    n_saves = None

    # console_log is the handle to a text file that records the console output
    if options.train:
        console_log = open(os.path.join(
            options_ex['model_folder'], "console_log.txt"), "w+")

    # lambda's are the ratios between the four error terms
    if not options.no_int:
        options_ex['lambda_class'] = 20  # 5
    else:
        options_ex['lambda_class'] = 1
    options_ex['lambda_ae'] = 1
    options_ex['lambda_p2s'] = 1
    options_ex['lambda_s2p'] = 1
    options_ex['lambda_div'] = 1
    options_ex['lambda_clst'] = 1
    options_ex['lambda_sep'] = 1

    # load prototypes if exist
    if not options.train and not options.no_int:
        options.n_prototypes = np.load(os.path.join(
            options_ex['model_folder'], 'prototypes.npy'), allow_pickle=True).shape[0]
        print("Assume: %s prototypes" % (options.n_prototypes))

    # create basic model architecture
    network_dict, tensor_dict = create_model_architecture(
        options.n_layers, dataset_dict, no_ae=options.no_ae)

    # complete model
    if not options.no_int:
        create_interpretable_model(
            options, dataset_dict, network_dict, tensor_dict)
        interpretable = 1
    else:
        create_blackbox_model(options, dataset_dict, network_dict, tensor_dict)
        interpretable = 0

    # Create the variable init operation and a saver object to store the model
    init = tf.global_variables_initializer()

    hyperparameters = {
        "learning_rate": options.learning_rate,
        "training_epochs": options.epochs,
        "batch_size": options.batch_size,
        "test_display_step": options.epochs // 10,
        "save_step": options.epochs // 10,
        "interpretable": interpretable,

        "lambda_class": options_ex['lambda_class'],
        "lambda_ae": options_ex['lambda_ae'],
        "lambda_p2s": options_ex['lambda_p2s'],
        "lambda_s2p": options_ex['lambda_s2p'],
        "lambda_div": options_ex['lambda_div'],
        "lambda_clst": options_ex['lambda_clst'],
        "lambda_sep": options_ex['lambda_sep'],

        "input_width": dataset_dict['input_width'],
        "n_input_channel": dataset_dict['n_input_channel'],
        "input_size": dataset_dict['input_width'] * dataset_dict['n_input_channel'],
        "n_classes": dataset_dict['n_classes'],

        "n_prototypes": options.n_prototypes,
        "n_layers": options.n_layers,

        "f_x":	network_dict['f_x'][0],

        "s_x": network_dict['stride_x'][0],

        "n_map_x": network_dict['n_map_x'][0],

        "n_features": network_dict['n_features'],
    }
    # save the hyperparameters above in the model snapshot
    for (name, value) in hyperparameters.items():
        tf.add_to_collection(
            'hyperparameters', tf.constant(name=name, value=value))
    saver = tf.train.Saver(max_to_keep=n_saves)

    # init variables
    sess.run(init)

    init_model(sess, saver, options, options_ex,
               dataset_dict, network_dict, tensor_dict)

    # process the blackbox
    if options.train and options.no_int:
        process_blackbox(sess, saver, console_log, options,
                         options_ex, dataset_dict, network_dict, tensor_dict)

    # process the interpretable
    if options.train and not options.no_int:
        process_interpretable(sess, saver, console_log, options,
                              options_ex, dataset_dict, network_dict, tensor_dict)

    # decode the prototypes
    if not options.skip_decode and (options.decode_latent_close or options.decode_latent):
        decoded_prototypes = decode_patch_prototypes(
            sess, options, dataset_dict, network_dict, tensor_dict)

        if options.decode_latent_close:
            decoded_prototypes_close = compute_prototypes_close(
                decoded_prototypes, options, dataset_dict, network_dict)

            plot_prototype_images(options.n_prototypes, decoded_prototypes_close,
                                  options_ex['img_folder'], 'prototypes_decoded_close')
            np.save(os.path.join(options_ex['model_folder'], 'prototypes_decoded_close.npy'),
                    decoded_prototypes_close)

        # decode prototypes no close
        if options.decode_latent:
            plot_prototype_images(options.n_prototypes, decoded_prototypes,
                                  options_ex['img_folder'], 'prototypes_decoded')
            np.save(os.path.join(options_ex['model_folder'], 'prototypes_decoded.npy'),
                    decoded_prototypes)

    # plot latent space comparison
    if options.plot_latent:
        _, best_prototype_patch, prot = compute_best_representative(
            sess, options, dataset_dict, network_dict, tensor_dict)
        plot_latent_prototypes(
            options.n_prototypes, dataset_dict['n_input_channel'], network_dict['n_map_x'], best_prototype_patch, prot)

    # compute quality of prototypes
    if options.quality:
        if not os.path.exists(os.path.join(options_ex['model_folder'], 'prototype_distances.npy')):
            best_prototype_distance, _, _ = compute_best_representative(
                sess, options, dataset_dict, network_dict, tensor_dict)
            np.save(os.path.join(
                options_ex['model_folder'], 'prototype_distances.npy'), best_prototype_distance)

        # load and print
        proto_qualities = np.load(os.path.join(
            options_ex['model_folder'], 'prototype_distances.npy'), allow_pickle=True)
        compute_prototype_distances(proto_qualities)

    # implement visualization
    if options.evaluate:
        options.build = True
        evaluate_and_build(sess, options, options_ex,
                           dataset_dict, network_dict, tensor_dict)


if __name__ == "__main__":
    # Command line options
    parser = OptionParser()

    # Data options
    parser.add_option("--data_path", action="store", type="string", dest="data_path",
                      default='data/character_trajectories/dataset_steps-20_timesteps-206.pickle', help="Data path")
    parser.add_option("--shuffle", action="store_true",
                      dest="shuffle", default=True, help="shuffle the dataset")
    parser.add_option("--seed", action="store", type="int",
                      dest="seed", default=0, help="random seed for any random operation")
    parser.add_option("--data_name", action="store", type="string",
                      dest="data_name", default='char', help="Dataset name, as it is not trivial to exctract it")

    # Model options
    parser.add_option("--model_name", action="store", type="string",
                      dest="model_name", default='test', help="Model name")
    parser.add_option("--n_prototypes", action="store", type="int",
                      dest="n_prototypes", default=40, help="Number prototypes")
    parser.add_option("--n_layers", action="store", type="int",
                      dest="n_layers", default=4, help="Number of layer defined in the model creation step")
    parser.add_option("--no_int", action="store_true",
                      dest="no_int", default=False, help="execute non interpretable model")
    parser.add_option("--no_ae", action="store_true",
                      dest="no_ae", default=False, help="execute without autoencoder model")

    # Prototype config
    parser.add_option("--predefine", action="store_true",
                      dest="predefine", default=True, help="predefine prototype classes (used to set force prototypes for each class)")  # depricated
    parser.add_option("--patch_size", action="store", type="int",
                      dest="patch_size", default=3, help="Latent space patch size")
    parser.add_option("--no_freeze", action="store_true",
                      dest="no_freeze", default=False, help="do not freeze layers, if set to true it does not freeze layers during prototype weight learning stage")  # depricated

    # Task options
    parser.add_option("--train", action="store_true",
                      dest="train", default=False, help="train model")
    parser.add_option("--evaluate", action="store_true",
                      dest="evaluate", default=False, help="evaluate model")
    parser.add_option("--build", action="store_true",
                      dest="build", default=False, help="reconstruct samples using prototypes")
    parser.add_option("--quality", action="store_true",
                      dest="quality", default=False, help="provides statistics of quality of prototypes")
    parser.add_option("--plot_latent", action="store_true",
                      dest="plot_latent", default=False, help="plot latent difference")
    parser.add_option("--decode_latent", action="store_true",
                      dest="decode_latent", default=False, help="decode of prototypes using the decoder part")
    parser.add_option("--decode_latent_close", action="store_true",
                      dest="decode_latent_close", default=False, help="decode of prototypes and find most sim to train set sampels")
    parser.add_option("--skip_decode", action="store_true",
                      dest="skip_decode", default=False, help="skips the creation of decode")
    parser.add_option("--no_plot", action="store_true",
                      dest="no_plot", default=False, help="skips plotting")

    # Trianing options
    parser.add_option("--learning_rate", action="store", type="float",
                      dest="learning_rate", default=0.002, help="learning rate")
    parser.add_option("--epochs", action="store", type="int",
                      dest="epochs", default=100, help="epochs")
    parser.add_option("--batch_size", action="store", type="int",
                      dest="batch_size", default=16, help="Batch size")

    # Evaluation options
    parser.add_option("--sort_val", action="store_true",
                      dest="sort_val", default=True, help="sorted by value otherwise sort by distance")  # depricated
    parser.add_option("--same_scale", action="store_true",
                      dest="same_scale", default=False, help="same scale for prototypes")

    # Build options
    parser.add_option("--build_mode", action="store", type="int",
                      dest="build_mode", default=0, help="best 20%, best value, worst value")
    parser.add_option("--build_cut", action="store", type="int",
                      dest="build_cut", default=100, help="cuts build after top X")
    parser.add_option("--show_distribution", action="store_true",
                      dest="show_distribution", default=False, help="show class distribution")
    parser.add_option("--full", action="store_true",
                      dest="full", default=False, help="show the full distribution")
    parser.add_option("--fill", action="store_true",
                      dest="fill", default=False, help="reconstruct sample and fill empty")
    parser.add_option("--full_build", action="store_true",
                      dest="full_build", default=False, help="seperate prototypes for patches")
    parser.add_option("--build_compare", action="store_true",
                      dest="build_compare", default=False, help="saves the builds to compare")

    # Parse command line options
    (options, args) = parser.parse_args()

    # print options
    print(options)

    process(options)
