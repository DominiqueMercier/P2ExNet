import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# sns.set()

#from modules.network_processor import get_se, get_patch, get_proto


def visualize_prototypes(epoch, n_prototypes, prototype_imgs, img_folder):
    """Visualizes the prototypes.

    Args:
        epoch (int): Current epoch.
        n_prototypes (int): Number of prototypes.
        prototype_imgs (list): List of prototypes.
        img_folder (string): Folder for the prototype image.
    """
    n_cols = int(np.min([5, np.sqrt(n_prototypes)]))
    n_rows = n_prototypes // n_cols + \
        1 if n_prototypes % n_cols != 0 else n_prototypes // n_cols
    g, b = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    for i in range(n_rows):
        for j in range(n_cols):
            if i*n_cols + j < n_prototypes:
                b[i][j].plot(prototype_imgs[i*n_cols + j])
                # b[i][j].axis('off')
            else:
                b[i][j].set_visible(False)
    plt.savefig(os.path.join(img_folder, 'prototype_result-' + str(epoch) + '.png'),
                # dpi=300,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()


def plot_build_org_comparison(dataset_dict, epoch, examples_to_show, encode_decode, img_folder):
    """Plot the comparison between the real series and the build that was created using the prototype.

    Args:
        dataset_dict (dict): dataset dictionary
        epoch (int): Current epoch.
        examples_to_show (array): Array with the indices to show.
        encode_decode (array): The decoded sample.
        img_folder (string): The image folder.
    """
    f, a = plt.subplots(2, examples_to_show,
                        figsize=(4*examples_to_show, 8))

    for i in range(examples_to_show):
        a[0][i].plot(dataset_dict['train_x'][i].reshape(
            dataset_dict['input_width'], dataset_dict['n_input_channel']))
        # a[0][i].axis('off')
        a[1][i].plot(encode_decode[i].reshape(
            dataset_dict['input_width'], dataset_dict['n_input_channel']))
        # a[1][i].axis('off')

    plt.savefig(os.path.join(img_folder, 'decoding_result-' + str(epoch) + '.png'),
                # dpi=300,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()


def plot_prototype_images(n_prototypes, decoded_prototypes, img_folder, name):
    """Plots the prototype images.

    Args:
        n_prototypes (int): Number of prototypes.
        decoded_prototypes_close (array): Decoded clostest prototypes.
        img_folder (string): Image folder.
    """
    n_cols = int(np.min([5, np.sqrt(n_prototypes)]))
    n_rows = n_prototypes // n_cols + \
        1 if n_prototypes % n_cols != 0 else n_prototypes // n_cols
    g, b = plt.subplots(
        n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    for i in range(n_rows):
        for j in range(n_cols):
            if i*n_cols + j < n_prototypes:
                b[i][j].plot(decoded_prototypes[i*n_cols + j])
                # b[i][j].axis('off')
            else:
                b[i][j].set_visible(False)
    plt.savefig(os.path.join(img_folder, name + '.png'),
                # dpi=300,
                bbox_inches='tight',
                pad_inches=0.1)
    plt.close()


def plot_latent_prototypes(n_prototypes, n_input_channel, n_map_x, prototype_patches, prot):
    """Plot the latent space comparison between the prototype and the representatvie patch.

    Args:
        n_prototypes (int): Number of prototypes.
        n_input_channel (int): Number of input channels.
        n_map_x (list): List with the filter numbers.
        prototype_patches (array): Best prototype patches.
        prot (array): Prototype feature vector.
    """
    for i in range(n_prototypes):
        p = np.reshape(prot[i], (n_input_channel, n_map_x[-1]))
        bp = np.reshape(
            prototype_patches[i], (n_input_channel, n_map_x[-1]))
        fig, ax = plt.subplots(1, n_input_channel, figsize=(16, 4))
        fig.suptitle('Prototype: ' + str(i))
        for j in range(n_input_channel):
            ax[j].plot(p[j])
            ax[j].plot(bp[j])
        plt.show()


def plot_class_distribution(options, sidx_dict):
    """Plots the class distribution for a sample.

    Args:
        options (dict): contains the options.
        sidx_dict (dict): sample parameters and data.
    """
    sidx_probabilites = sidx_dict['sidx_w'] * \
        np.transpose(sidx_dict['w_squeeze'], [1, 0])

    class_dist = np.sum(sidx_probabilites, axis=-1)
    class_dist = class_dist - np.min(class_dist)

    cm = plt.get_cmap('tab20')
    cNorm = matplotlib.colors.Normalize(
        vmin=0, vmax=len(class_dist)-1)
    scalarMap = matplotlib.cm.ScalarMappable(
        norm=cNorm, cmap=cm)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax_flat = ax.flat
    fig.suptitle('Sample: %s | GT: %s | Pred: %s' % (
        sidx_dict['sidx_num'], sidx_dict['sidx_gt'], sidx_dict['sidx_pred']))
    ax_flat[0].plot(sidx_dict['sidx_x'])
    ax_flat[1].set_prop_cycle(
        color=[scalarMap.to_rgba(k) for k in range(len(class_dist))])

    ax_flat[1].xaxis.set_major_locator(
        MaxNLocator(integer=True))
    for c in range(len(class_dist)):
        ax_flat[1].bar(
            c, class_dist[c], label='Class ' + str(c))

    handles, labels = ax_flat[1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(
        0.5, 0.95), loc='upper center', ncol=np.min([8, len(class_dist)]))
    plt.subplots_adjust(top=0.8)
    plt.show()


def plot_class_distribution_full(options, dataset_dict, network_dict, sidx_dict):
    """Plots the class distribution full for a sample.

    Args:
        options (dict): contains the options.
        dataset_dict (dict): dataset parameters.
        network_dict (dict): network parameters.
        sidx_dict (dict): sample parameters and data.
    """
    from modules.network_processor import get_se

    sidx_probabilites = sidx_dict['sidx_w'] * \
        np.transpose(sidx_dict['w_squeeze'], [1, 0])

    sidx_probabilites = np.reshape(
        sidx_probabilites, [dataset_dict['n_classes'], network_dict['n_patches'], options.n_prototypes])

    data = np.max(sidx_probabilites, axis=-1)
    data -= np.min(data, axis=0)
    patch_classes = np.arange(data.shape[1])

    n_plots = data.shape[1] * 2
    n_cols = int(np.min([6, np.sqrt(n_plots)]))
    n_rows = n_plots // n_cols + \
        1 if n_plots % n_cols != 0 else n_plots // n_cols

    cm = plt.get_cmap('tab20')
    cNorm = matplotlib.colors.Normalize(
        vmin=0, vmax=dataset_dict['n_classes']-1)
    scalarMap = matplotlib.cm.ScalarMappable(
        norm=cNorm, cmap=cm)

    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    ax_flat = ax.flat
    fig.suptitle('Sample: %s | GT: %s | Pred: %s' % (
        sidx_dict['sidx_num'], sidx_dict['sidx_gt'], sidx_dict['sidx_pred']))

    # create fake labels
    for i in patch_classes:
        ax_flat[1].bar(i, np.zeros(len(patch_classes)),
                       label='Class ' + str(i))
    handles, labels = ax_flat[1].get_legend_handles_labels()

    for i in range(n_plots):
        i_real = i // 2
        if i_real < data.shape[1] and i % 2 == 0:
            ax_flat[i].plot(sidx_dict['sidx_x'])
            se = get_se(
                i_real, network_dict['start'], network_dict['jump_size'], network_dict['rf_size'], options.patch_size, dataset_dict['input_width'])
            ax_flat[i].axvline(x=se[0], color='k')
            ax_flat[i].axvline(x=se[1], color='k')
        elif i_real < data.shape[1] and i % 2 == 1:
            ax_flat[i].set_prop_cycle(
                color=[scalarMap.to_rgba(k) for k in range(len(patch_classes))])
            ax_flat[i].xaxis.set_major_locator(
                MaxNLocator(integer=True))

            ax_flat[i].set_ylim(0, np.max(data))
            ax_flat[i].bar(np.arange(data.shape[0]), data[:, i_real], color=[
                           scalarMap.to_rgba(k) for k in range(len(patch_classes))])
        else:
            ax_flat[i].set_visible(False)

    fig.legend(handles, labels, bbox_to_anchor=(
        0.5, 0.95), loc='upper center', ncol=np.min([8, len(patch_classes)]))

    plt.close()


def visualize_prototype_images(options, dataset_dict, network_dict, sidx_dict, prototype_adjusted_imgs):
    """Visualizes the sample with the replaced parts.

    Args:
        options (dict): contains the options.
        dataset_dict (dict): dataset parameters.
        network_dict (dict): network parameters.
        sidx_dict (dict): sample parameters and data.
        prototype_adjusted_imgs (array): image adjusted to the corresponding patch and prototype.
    """
    from modules.network_processor import get_patch, get_proto, get_se

    # visualize the prototype images
    n_cols = int(np.min([5, np.sqrt(options.n_prototypes)]))
    n_rows = (options.n_prototypes + 2) // n_cols + 1 if (options.n_prototypes +
                                                          2) % n_cols != 0 else (options.n_prototypes + 2) // n_cols
    n_rows = np.min([n_rows, 10])
    fig, ax = plt.subplots(
        n_rows, n_cols, figsize=(4*n_cols, 4*n_rows), sharex=True, sharey=options.same_scale)
    ax_flat = ax.flat
    fig.suptitle('Sample: %s | GT: %s | Pred: %s' % (
        sidx_dict['sidx_num'], sidx_dict['sidx_gt'], sidx_dict['sidx_pred']))
    ax_flat[0].plot(sidx_dict['sidx_x'])
    ax_flat[0].tick_params(
        labelbottom=True, labelleft=True)
    ax_off = 2 if options.build else 1

    per_class = int(np.ceil(options.n_prototypes / dataset_dict['n_classes']))
    # plots for the prototypes
    for aidx, iAx in enumerate(ax_flat):
        # input and build
        if aidx == 0 or (aidx == 1 and options.build):
            continue

        # prototypes
        ax_pidx = aidx - ax_off  # position in plot
        if ax_pidx < options.n_prototypes:
            pidx = sidx_dict['sidx_w_argsort'][ax_pidx]
            s, e = get_se(
                sidx_dict['sidx_ind'][pidx] if not options.full_build else get_patch(
                    pidx, options.n_prototypes), network_dict['start'], network_dict['jump_size'],
                network_dict['rf_size'], options.patch_size, dataset_dict['input_width'])
            e -= 1  # index shift by 1

            # title
            c = 'black'
            if sidx_dict['sidx_w'][pidx] > 0:
                c = 'red'
            if ((pidx if not options.full_build else get_proto(pidx, options.n_prototypes)) // per_class) == sidx_dict['sidx_gt']:
                c = 'green'
            iAx.set_title('Pid: %s | Val: %s | Dis: %s | S: %s | E: %s' % (
                pidx if not options.full_build else get_proto(pidx, options.n_prototypes), int(
                    100*sidx_dict['sidx_w'][pidx])/100, int(100*sidx_dict['sidx_dist'][pidx])/100,
                s, e), color=c)

            # prototype
            x_range = np.arange(
                s, prototype_adjusted_imgs[pidx].shape[0] + s)
            x_range = np.stack(
                [x_range for i in range(dataset_dict['n_input_channel'])], axis=1)

            iAx.plot(
                x_range, prototype_adjusted_imgs[pidx])
            iAx.tick_params(
                labelbottom=True, labelleft=True)
            if pidx in sidx_dict['sidx_build_replaced']:
                iAx.set_facecolor('yellow')

        else:
            iAx.set_visible(False)

    # build plot
    if options.build:
        # vertical replacements
        for se in sidx_dict['sidx_build_se_list']:
            ax_flat[1].axvspan(
                se[0], se[1], alpha=0.5, color='yellow')
            #ax_flat[1].axvline(x=se[0], color='k')
            #ax_flat[1].axvline(x=se[1], color='k')

        ax_flat[1].tick_params(
            labelbottom=True, labelleft=True)
        ax_flat[1].plot(sidx_dict['sidx_build'])
        c = 'green' if sidx_dict['sidx_pred'] == sidx_dict['build_pred'] else 'red'
        ax_flat[1].set_title('Pred: %s | Dis: %s' % (
            sidx_dict['build_pred'], sidx_dict['sidx_build_dif']), color=c)

    plt.subplots_adjust(hspace=0.8)
    # plt.savefig(os.path.join(img_folder, 'prototype_result-' + str(sample_num) + '.png'),
    #             #dpi=300,
    #             bbox_inches='tight',
    #             pad_inches=0.1)
    plt.close()
    #plt.show()
