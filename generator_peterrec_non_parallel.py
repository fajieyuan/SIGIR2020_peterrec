import tensorflow as tf
import ops
import modeling
import numpy as np

class NextItNet_Decoder:

    def __init__(self, model_para):
        self.model_para = model_para
        self.embedding_width =  model_para['dilated_channels']
        self.allitem_embeddings = tf.get_variable('allitem_embeddings',
                                                    [model_para['item_size'], self.embedding_width],
                                                    initializer=tf.truncated_normal_initializer(stddev=0.02))
        self.itemseq_input = tf.placeholder('int32',
                                            [None, None], name='itemseq_input')
        self.masked_position = tf.placeholder('int32',
                                              [None, None], name='masked_position')

    def train_graph(self, cardinality=32,mp=False,is_negsample=False):
        self.masked_items = tf.placeholder('int32',
                                           [None, None], name='masked_items')
        self.label_weights = tf.placeholder(tf.float32,
                                            [None, None], name='label_weights')
        self.dilate_input=self.model_graph(self.itemseq_input,train=True,mp=mp,cardinality=cardinality)
        self.softmax_w = tf.get_variable("softmax_w", [self.model_para['item_size'], self.embedding_width], tf.float32,tf.random_normal_initializer(0.0, 0.01))

    def model_graph(self, itemseq_input,train,mp,cardinality):
        model_para = self.model_para
        self.context_embedding = tf.nn.embedding_lookup(self.allitem_embeddings,
                                                   itemseq_input, name="context_embedding")
        if self.model_para['has_positionalembedding']:
            pos_emb = self.embedding(
                tf.tile(tf.expand_dims(tf.range(tf.shape(itemseq_input)[1]), 0),
                        [tf.shape(self.itemseq_input)[0], 1]),
                max_position=model_para['max_position'],
                num_units=self.embedding_width,
                zero_pad=False,
                scale=False,
                l2_reg=0.0,
                scope="dec_pos",
                with_t=False
            )
            dilate_input = tf.concat([ self.context_embedding, pos_emb], -1)
        else:
            dilate_input = self.context_embedding
        residual_channels = dilate_input.get_shape().as_list()[-1]
        for layer_id, dilation in enumerate(model_para['dilations']):
            dilate_input = ops.peter_2mp_parallel(dilate_input, dilation,
                                                        layer_id, residual_channels,
                                                        model_para['kernel_size'], causal=False, train=train,mp=mp,cardinality=cardinality)

        return dilate_input

    def gather_indexes(self,sequence_tensor, positions):
        sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]
        flat_offsets = tf.reshape(
            tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        flat_positions = tf.reshape(positions + flat_offsets, [-1])
        flat_sequence_tensor = tf.reshape(sequence_tensor,
                                          [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
        return output_tensor

    def embedding(self, inputs, max_position, num_units,zero_pad=True,scale=True,l2_reg=0.0, scope="embedding", with_t=False):
        with tf.variable_scope(scope):
            lookup_table = tf.get_variable('lookup_table_position',
                                           dtype=tf.float32,
                                           shape=[max_position, num_units],
                                           # initializer=tf.contrib.layers.xavier_initializer(),
                                           regularizer=tf.contrib.layers.l2_regularizer(l2_reg))
            if zero_pad:
                lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                          lookup_table[1:, :]), 0)
            outputs = tf.nn.embedding_lookup(lookup_table, inputs)
            if scale:
                outputs = outputs * (num_units ** 0.5)
        if with_t:
            return outputs, lookup_table
        else:
            return outputs










