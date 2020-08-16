import tensorflow as tf
import ops
import modeling
import numpy as np


"almost the same with generator_recsys_cloze.py, use self.dilate_input so that we can use tf.get_collection"

class NextItNet_Decoder:

    def __init__(self, model_para):

        self.model_para = model_para
        self.is_negsample = model_para['is_negsample']
        self.embedding_width =  model_para['dilated_channels']
        self.allitem_embeddings = tf.get_variable('allitem_embeddings',
                                                    [model_para['item_size'], self.embedding_width],
                                                    initializer=tf.truncated_normal_initializer(stddev=0.02))

        self.itemseq_input = tf.placeholder('int32',
                                            [None, None], name='itemseq_input')
        self.masked_position = tf.placeholder('int32',
                                              [None, None], name='masked_position')


        self.softmax_w = tf.get_variable("softmax_w", [self.model_para['item_size'], self.embedding_width], tf.float32,
                                         tf.random_normal_initializer(0.0, 0.01))
        self.softmax_b = tf.get_variable(
            "softmax_b",
            shape=[self.model_para['item_size']],
            initializer=tf.constant_initializer(0.1))



    def train_graph(self):

        self.masked_items = tf.placeholder('int32',
                                           [None, None], name='masked_items')
        self.label_weights = tf.placeholder(tf.float32,
                                            [None, None], name='label_weights')
        self.dilate_input=self.model_graph(self.itemseq_input,train=True)
        self.loss = self.get_masked_lm_output(self.model_para, self.dilate_input, self.softmax_w, self.masked_position,
                                              self.masked_items, self.label_weights, trainable=True)

    def model_graph(self, itemseq_input,train):
        model_para = self.model_para


        self.context_embedding = tf.nn.embedding_lookup(self.allitem_embeddings,
                                                   itemseq_input, name="context_embedding")

        #positional embedding

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


            dilate_input = ops.nextitnet_residual_block(dilate_input, dilation,
                                                        layer_id, residual_channels,
                                                        model_para['kernel_size'], causal=False, train=train)


        return dilate_input




    def predict_graph(self,reuse=False,is_negsample=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()


        dilate_input = self.model_graph(self.itemseq_input, train=False)

        model_para = self.model_para

        if self.is_negsample:
            logits_2D = tf.reshape(dilate_input[:, -1:, :], [-1, self.embedding_width])
            logits_2D = tf.matmul(logits_2D, self.softmax_w, transpose_b=True)
            logits_2D = tf.nn.bias_add(logits_2D, self.softmax_b)
        else:
            logits = ops.conv1d(tf.nn.relu(dilate_input)[:, -1:, :], model_para['item_size'], name='logits')
            logits_2D = tf.reshape(logits, [-1, model_para['item_size']])
        probs_flat = tf.nn.softmax(logits_2D)
        # self.g_probs = tf.reshape(probs_flat, [-1, tf.shape(self.input_predict)[1], model_para['item_size']])
        self.log_probs = probs_flat
        self.top_k = tf.nn.top_k(probs_flat, k=model_para['top_k'], name='top-k')


    def gather_indexes(self,sequence_tensor, positions):
        """Gathers the vectors at the specific positions over a minibatch."""
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


    def get_masked_lm_output(self, bert_config, input_tensor, output_weights, positions,
                             label_ids, label_weights,trainable):
        """Get loss and log probs for the masked LM."""
        input_tensor = self.gather_indexes(input_tensor, positions)



        if self.is_negsample:
            logits_2D = input_tensor
            label_flat = tf.reshape(label_ids, [-1, 1])  # 1 is the number of positive example
            num_sampled = int(0.2 * self.model_para['item_size'])  # sample 20% as negatives
            loss = tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, label_flat, logits_2D,
                                              num_sampled,
                                              self.model_para['item_size'])
        else:
            sequence_shape = modeling.get_shape_list(positions)
            batch_size = sequence_shape[0]
            seq_length = sequence_shape[1]
            residual_channels = input_tensor.get_shape().as_list()[-1]
            input_tensor = tf.reshape(input_tensor, [-1, seq_length, residual_channels])

            logits = ops.conv1d(tf.nn.relu(input_tensor), self.model_para['item_size'], name='logits')
            logits_2D = tf.reshape(logits, [-1, self.model_para['item_size']])
            label_flat = tf.reshape(label_ids, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flat, logits=logits_2D)
        loss = tf.reduce_mean(loss)

        #not sure the impact, 0.001 is empirical value
        # regularization = 0.001 * tf.reduce_mean([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        # loss=loss+regularization
        return loss




    #item_size means the maximum size of the sequence the code is from Self-Attentive Sequential Recommendation
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










