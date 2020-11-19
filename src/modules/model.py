import tensorflow as tf


def create_model_architecture(n_layers, dataset_dict, no_ae=False):
    """This methods creates the basic model architecture used for both the interpretabel and the non interpretable model. 
    You can remoce the decoder part as well.

    Args:
        n_layers (int): Number of layers for each part of the structure. If decoder is included this is the number of layers for the encoder.
        dataset_dict (dict): Contains all dataset parameters.
        no_ae (bool, optional): Removes the autoencoder behavior by excluding the decoder. Keeps the encoder. Defaults to False.

    Returns:
        network_dict (dict): Dictionary that has all network parameters.
        tensor_dict (dict): Dictionary that has all tensors.
    """
    input_size = dataset_dict['input_width'] * dataset_dict['n_input_channel']

    # width of each layers' filters
    f_x = [3 for i in range(n_layers)]

    # stride size in each direction for each of the layers
    stride_x = [2 for i in range(n_layers)]

    # number of feature maps in each layer
    n_map_x = [32 for i in range(n_layers)]

    # the shapes of each layer's filter
    filter_shape_x = []
    for i in range(n_layers):
        if i == 0:
            filter_shape_x.append(
                [f_x[0], dataset_dict['n_input_channel'], n_map_x[0]])
        else:
            filter_shape_x.append([f_x[i], n_map_x[i-1], n_map_x[i]])

    # tf Graph input
    # X is the 2-dimensional matrix whose every row is an image example.
    # Y is the 2-dimensional matrix whose every row is the one-hot encoding label.
    X = tf.placeholder(dtype=tf.float32, shape=[None, input_size], name='X')
    X_img = tf.reshape(
        X, shape=[-1, dataset_dict['input_width'], dataset_dict['n_input_channel']], name='X_img')
    Y = tf.placeholder(dtype=tf.float32, shape=[
                       None, dataset_dict['n_classes']], name='Y')

    # We create a tf placeholder for every lambda so that they can be tweaked during training
    lambda_class_t = tf.placeholder(
        dtype=tf.float32, shape=(), name="lambda_class_t")
    lambda_ae_t = tf.placeholder(
        dtype=tf.float32, shape=(), name="lambda_ae_t")
    lambda_p2s_t = tf.placeholder(
        dtype=tf.float32, shape=(), name="lambda_p2s_t")
    lambda_s2p_t = tf.placeholder(
        dtype=tf.float32, shape=(), name="lambda_s2p_t")
    lambda_div_t = tf.placeholder(
        dtype=tf.float32, shape=(), name="lambda_div_t")
    lambda_clst_t = tf.placeholder(
        dtype=tf.float32, shape=(), name="lambda_clst_t")
    lambda_sep_t = tf.placeholder(
        dtype=tf.float32, shape=(), name="lambda_sep_t")

    # create the weights and bias variables
    weights = {}
    biases = {}
    initializer = tf.contrib.layers.xavier_initializer()
    for i in range(n_layers):
        i_rev = n_layers - (i+1)
        num_enc = str(i+1)
        num_dec = str(n_layers - i)
        weights['enc_f' + num_enc] = tf.Variable(initializer(
            shape=filter_shape_x[i]), name='encoder_f' + num_enc)
        biases['enc_b' + num_enc] = tf.Variable(tf.zeros([n_map_x[i]], dtype=tf.float32),
                                                name='encoder_b' + num_enc)
        if not no_ae:
            weights['dec_f' + num_dec] = tf.Variable(initializer(
                shape=filter_shape_x[i_rev]), name='decoder_f' + num_dec)
            if i_rev > 0:
                biases['dec_b' + num_dec] = tf.Variable(tf.zeros([n_map_x[i_rev]], dtype=tf.float32),
                                                        name='decoder_b' + num_dec)
            else:
                biases['dec_b' + num_dec] = tf.Variable(tf.zeros([dataset_dict['n_input_channel']], dtype=tf.float32),
                                                        name='decoder_b' + num_dec)

    # construct the model
    # eln means the output of the nth layer of the encoder
    # we compute the output shape of each layer because the deconv_layer function requires it
    el_x = []
    lx_shape = []
    for i in range(n_layers):
        num = str(i + 1)
        if i == 0:
            el = conv_layer(X_img, weights['enc_f' + num],
                            biases['enc_b' + num], stride_x[i], "SAME")
            el_x.append(el)
        else:
            el = conv_layer(el_x[-1], weights['enc_f' + num],
                            biases['enc_b' + num], stride_x[i], "SAME")
            el_x.append(el)
        lx_shape.append(el.get_shape().as_list())

    # create dictionaries
    network_dict = {}
    network_dict['f_x'] = f_x
    network_dict['stride_x'] = stride_x
    network_dict['n_map_x'] = n_map_x
    network_dict['el_x'] = el_x
    network_dict['lx_shape'] = lx_shape

    tensor_dict = {}
    tensor_dict['X'] = X
    tensor_dict['X_img'] = X_img
    tensor_dict['Y'] = Y
    tensor_dict['weights'] = weights
    tensor_dict['biases'] = biases
    tensor_dict['lambda_class_t'] = lambda_class_t
    tensor_dict['lambda_ae_t'] = lambda_ae_t
    tensor_dict['lambda_p2s_t'] = lambda_p2s_t
    tensor_dict['lambda_s2p_t'] = lambda_s2p_t
    tensor_dict['lambda_div_t'] = lambda_div_t
    tensor_dict['lambda_clst_t'] = lambda_clst_t
    tensor_dict['lambda_sep_t'] = lambda_sep_t

    return network_dict, tensor_dict

########################################
# basic implementation of conv_layer,
# decond_layer, and fc_layer parameters
# and returns are straight forward
########################################


def conv_layer(input, filter, bias, strides, padding="VALID", nonlinearity=tf.nn.relu):
    conv = tf.nn.conv1d(input, filter, stride=strides, padding=padding)
    #act = nonlinearity(conv + bias)
    act = conv + bias
    return act


def deconv_layer(input, filter, bias, output_shape, strides, padding="VALID", nonlinearity=tf.nn.relu):
    deconv = tf.contrib.nn.conv1d_transpose(
        input, filter, output_shape, strides, padding=padding)
    #act = nonlinearity(deconv + bias)
    act = deconv + bias
    return act


def fc_layer(input, weight, bias, nonlinearity=tf.nn.relu):
    return nonlinearity(tf.matmul(input, weight) + bias)
