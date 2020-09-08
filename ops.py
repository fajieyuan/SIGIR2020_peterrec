import tensorflow as tf
import math
import numpy as np



#config e.g. dilations: [1,4,16,] In most cases[1,4,] is enough
def nextitnet_residual_block(input_, dilation, layer_id,
                            residual_channels, kernel_size,
                            causal=True, train=True):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):

        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)


        dilated_conv = conv1d(relu1,  residual_channels,
                              2 *dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        return input_ + relu1

#Aggregated Residual Transformations for Deep Neural Networks block1  =resnet if cardinality==1
def get_mp(input_,cardinality=32, name="mp"):
    with tf.variable_scope(name):
        residual_channels = input_.get_shape()[-1]
        hidden_size = residual_channels / (cardinality * 4)
        blocksets = list()
        for i in range(cardinality):
            conv_down_i = conv1d(input_, hidden_size,
                               name="mp_conv1_down_{}".format(i)
                               )
            conv_down_i = gelu(conv_down_i)
            conv_up_i = conv1d(conv_down_i, residual_channels,
                             name="mp_conv1_up_{}".format(i)
                             )
            blocksets.append(conv_up_i)

        output = tf.add_n(blocksets)
        return input_+output


# peter_2mp_parallel
def peter_2mp_parallel(input_, dilation, layer_id,
                            residual_channels, kernel_size,
                            causal=True, train=True,mp=True,cardinality=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):

        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        if mp:
            after_adapter = get_mp(input_, cardinality,name="mp_1")
            dilated_conv = tf.add(dilated_conv, after_adapter)


        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        #input_ln=tf.contrib.layers.layer_norm(dilated_conv,reuse=not train, trainable=train)  #performance is not good, paramter wrong?
        relu1 = tf.nn.relu(input_ln)


        dilated_conv = conv1d(relu1,  residual_channels,
                              2 *dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        if mp:
            after_adapter=get_mp(relu1,cardinality,name="mp_2")
            dilated_conv = tf.add(dilated_conv, after_adapter)


        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        return relu1+input_


# peter_2mp_parallel  peter_2mp_serial
def peter_2mp_serial(input_, dilation, layer_id,
                            residual_channels, kernel_size,
                            causal=True, train=True,mp=True,cardinality=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):
        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        if mp:
            after_adapter = get_mp(dilated_conv, cardinality,name="mp_1")
            dilated_conv = after_adapter
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        dilated_conv = conv1d(relu1,  residual_channels,
                              2 *dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )
        if mp:
            after_adapter=get_mp(dilated_conv,cardinality,name="mp_2")
            dilated_conv = after_adapter
        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)
        return input_ + relu1


def peter_mp_serial(input_, dilation, layer_id,
                            residual_channels, kernel_size,
                            causal=True, train=True,mp=True,cardinality=32):
    resblock_type = "decoder"
    resblock_name = "nextitnet_residual_block{}_layer_{}_{}".format(resblock_type, layer_id, dilation)
    with tf.variable_scope(resblock_name,reuse=tf.AUTO_REUSE):

        dilated_conv = conv1d(input_, residual_channels,
                              dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv1"
                              )
        input_ln = layer_norm(dilated_conv, name="layer_norm1", trainable=train)
        relu1 = tf.nn.relu(input_ln)


        dilated_conv = conv1d(relu1,  residual_channels,
                              2 *dilation, kernel_size,
                              causal=causal,
                              name="dilated_conv2"
                              )

        if mp:
            after_adapter=get_mp(dilated_conv,cardinality)
            dilated_conv = after_adapter


        input_ln = layer_norm(dilated_conv, name="layer_norm2", trainable=train)
        relu1 = tf.nn.relu(input_ln)

        return input_ + relu1


def conv1d(input_, output_channels,
           dilation=1, kernel_size=1, causal=False,
           name="dilated_conv"):
    with tf.variable_scope(name):
        weight = tf.get_variable('weight', [1, kernel_size, input_.get_shape()[-1], output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=0.02, seed=1))
        bias = tf.get_variable('bias', [output_channels],
                               initializer=tf.constant_initializer(0.0))

        if causal:
            padding = [[0, 0], [(kernel_size - 1) * dilation, 0], [0, 0]]
            padded = tf.pad(input_, padding)
            input_expanded = tf.expand_dims(padded, dim=1)
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='VALID') + bias
        else:
            input_expanded = tf.expand_dims(input_, dim=1)
            # out = tf.nn.conv2d(input_expanded, weight, strides=[1, 1, 1, 1], padding="SAME") + bias
            out = tf.nn.atrous_conv2d(input_expanded, weight, rate=dilation, padding='SAME') + bias


        return tf.squeeze(out, [1])


# tf.contrib.layers.layer_norm
def layer_norm(x, name, epsilon=1e-8, trainable=True):
    with tf.variable_scope(name):
        shape = x.get_shape()
        beta = tf.get_variable('beta', [int(shape[-1])],
                               initializer=tf.constant_initializer(0), trainable=trainable)
        gamma = tf.get_variable('gamma', [int(shape[-1])],
                                initializer=tf.constant_initializer(1), trainable=trainable)

        mean, variance = tf.nn.moments(x, axes=[len(shape) - 1], keep_dims=True)

        x = (x - mean) / tf.sqrt(variance + epsilon)

        return gamma * x + beta

def gelu(x):
        """Gaussian Error Linear Unit.

        This is a smoother version of the RELU.
        Original paper: https://arxiv.org/abs/1606.08415
        Args:
          x: float Tensor to perform activation.

        Returns:
          `x` with the GELU activation applied.
        """
        cdf = 0.5 * (1.0 + tf.tanh(
            (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
        return x * cdf
