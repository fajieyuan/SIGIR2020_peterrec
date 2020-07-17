import tensorflow as tf
import ops
import numpy as np

class NextItNet_Decoder:

    def __init__(self, model_para):
        self.model_para = model_para
        embedding_width =  model_para['dilated_channels']
        self.allitem_embeddings = tf.get_variable('allitem_embeddings',
                                                    [model_para['item_size'], embedding_width],
                                                    initializer=tf.truncated_normal_initializer(stddev=0.02))

    def train_graph(self, is_negsample=False):
        self.itemseq_input = tf.placeholder('int32',
                                         [None, None], name='itemseq_input')
        label_seq, self.dilate_input=self.model_graph(self.itemseq_input, train=True)

        model_para = self.model_para
        if is_negsample:
            logits_2D = tf.reshape(self.dilate_input, [-1,model_para['dilated_channels']])
            self.softmax_w = tf.get_variable("softmax_w", [model_para['item_size'],  model_para['dilated_channels']],tf.float32,tf.random_normal_initializer(0.0, 0.01))
            self.softmax_b = tf.get_variable("softmax_b", [model_para['item_size']], tf.float32, tf.constant_initializer(0.1))
            label_flat = tf.reshape(label_seq, [-1, 1])  # 1 is the number of positive example
            num_sampled = int(0.2* model_para['item_size'])#sample 20% as negatives
            loss =tf.nn.sampled_softmax_loss(self.softmax_w, self.softmax_b, label_flat, logits_2D, num_sampled,model_para['item_size'])

        else:
            logits = ops.conv1d(tf.nn.relu(self.dilate_input), model_para['item_size'], name='logits')
            logits_2D = tf.reshape(logits, [-1, model_para['item_size']])
            label_flat = tf.reshape(label_seq, [-1])
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flat, logits=logits_2D)

        self.loss = tf.reduce_mean(loss)
        self.arg_max_prediction = tf.argmax(logits_2D, 1) #useless, if using negative sampling (i.e., negsample=True), it should be changed such as in predict_graph module



    def model_graph(self, itemseq_input, train=True):
        model_para = self.model_para
        context_seq = itemseq_input[:, 0:-1]
        label_seq = itemseq_input[:, 1:]

        self.context_embedding = tf.nn.embedding_lookup(self.allitem_embeddings,
                                                   context_seq, name="context_embedding")
        dilate_input = self.context_embedding
        for layer_id, dilation in enumerate(model_para['dilations']):
            dilate_input = ops.nextitnet_residual_block(dilate_input, dilation,
                                                        layer_id, model_para['dilated_channels'],
                                                        model_para['kernel_size'], causal=True, train=train)
        return label_seq, dilate_input



    def predict_graph(self, is_negsample=False, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        self.input_predict = tf.placeholder('int32', [None, None], name='input_predict')

        label_seq, dilate_input = self.model_graph(self.input_predict, train=False)
        model_para = self.model_para

        if is_negsample:
            logits_2D = tf.reshape(dilate_input[:, -1:, :], [-1, model_para['dilated_channels']])
            logits_2D = tf.matmul(logits_2D, tf.transpose(self.softmax_w))
            logits_2D = tf.nn.bias_add(logits_2D, self.softmax_b)
        else:
            logits = ops.conv1d(tf.nn.relu(dilate_input[:, -1:, :]), model_para['item_size'], name='logits')
            logits_2D = tf.reshape(logits, [-1, model_para['item_size']])

        label_flat = tf.reshape(label_seq[:, -1], [-1])
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_flat, logits=logits_2D)
        self.loss_test = tf.reduce_mean(loss)
        probs_flat = tf.nn.softmax(logits_2D)
        self.g_probs = tf.reshape(probs_flat, [-1, 1, model_para['item_size']]) #slow 
        self.top_k = tf.nn.top_k(self.g_probs[:, -1], k=5, name='top-k') #faster, if you change [probs] = sess.run([itemrec.g_probs],....) in NextitNet_TF_Pretrain.py to top_k=sess.run([itemrec.top_k],....) 
























