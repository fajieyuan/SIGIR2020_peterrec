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

    def train_graph(self, cardinality=32,mp=False):
        self.itemseq_input = tf.placeholder('int32',
                                         [None, None], name='itemseq_input')
        self.label_seq, self.dilate_input=self.model_graph(self.itemseq_input, train=True,mp = mp, cardinality=cardinality)

    def model_graph(self, itemseq_input,train,mp,cardinality):
        model_para = self.model_para
        # for finetuning purpose, input is like 1 2 3 4 5 [CLS] 134, note 134 is target label
        context_seq = itemseq_input[:, 0:-1]# 1 2 3 4 5 [CLS]
        label_seq = itemseq_input[:, -1:]# 134
        context_embedding = tf.nn.embedding_lookup(self.allitem_embeddings,
                                                   context_seq, name="context_embedding")
        dilate_input = context_embedding
        residual_channels = dilate_input.get_shape().as_list()[-1]
        for layer_id, dilation in enumerate(model_para['dilations']):
            #  dilate_input=ops.peter_2mp_serial also performs well even with one mp
            dilate_input = ops.peter_2mp_serial(dilate_input, dilation,
                                                                     layer_id, residual_channels,
                                                                     model_para['kernel_size'], causal=True,
                                                                     train=train, mp=mp,
                                                                     cardinality=cardinality)

        return label_seq, dilate_input






