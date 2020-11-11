import tensorflow as tf
import data_loader_recsys_transfer_finetune as data_loader_recsys
import generator_peterrec_cau_parallel as generator_recsys
import utils
import shutil
import time
import math
import eval
import numpy as np
import argparse
import sys
import ops

#cau-->causal cnn

def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t == s:
        t = np.random.randint(l, r)
    return t

def random_negs(l,r,no,s):
    # set_s=set(s)
    negs = []
    for i in range(no):
        t = np.random.randint(l, r)
        # while (t in set_s):
        while (t== s):
            t = np.random.randint(l, r)
        negs.append(t)
    return negs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--top_k', type=int, default=5,
                        help='Sample from top k predictions')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--eval_iter', type=int, default=1000,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=1000,
                        help='save model parameters every')
    parser.add_argument('--datapath', type=str, default='Data/Session/history_sequences_20181014_fajie_transfer_finetune_small.csv',
                        help='data path')
    parser.add_argument('--tt_percentage', type=float, default=0.9,
                        help='default=0.2 means 80% training 20% testing')
    parser.add_argument('--is_generatesubsession', type=bool, default=False,
                        help='whether generating a subsessions, e.g., 12345-->01234,00123,00012  It may be useful for very some very long sequences')
    parser.add_argument('--has_positionalembedding', type=bool, default=False,
                        help='whether contains positional embedding before performing cnnn')

    parser.add_argument('--padtoken', type=str, default='0',
                        help='is the padding token in the beggining of the sequence')
    parser.add_argument('--negtive_samples', type=int, default='99',
                        help='the number of negative examples for each positive one')

    args = parser.parse_args()



    dl = data_loader_recsys.Data_Loader({'model_type': 'generator', 'dir_name': args.datapath})

    all_samples = dl.example

    items = dl.item_dict
    items_len = len(items)
    print "len(items)", len(items)
    targets = dl.target_dict
    targets_len=len(targets)
    print "len(targets)", len(targets)

    negtive_samples=args.negtive_samples
    top_k=args.top_k

    if items.has_key(args.padtoken):
        padtoken = items[args.padtoken]  # is the padding token in the beggining of the sentence
    else:
        padtoken=len(items)+1

    # Randomly shuffle data you cannnot change  np.random.seed(10) unless you change it in  nextitred.py
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(all_samples)))
    all_samples = all_samples[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
    train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]

    model_para = {
        #all parameters shuold be consist with those in NextitNet_TF_Pretrain.py!!!!
        'item_size': len(items),
        'target_item_size': len(targets),
        'dilated_channels': 64,
        'cardinality': 1, #using a large number does not performs better. cardinality=1 denotes the standard residual block
        'dilations': [1,4,1,4,1,4,1,4,],
        'kernel_size': 3,
        'learning_rate':0.001,
        'batch_size':3,  #you can not use batch_size=1 since we use np.squeeze will reuduce one dimension
        'iterations': 400,
        'has_positionalembedding': args.has_positionalembedding
    }


    itemrec = generator_recsys.NextItNet_Decoder(model_para)
    itemrec.train_graph(cardinality=model_para['cardinality'],mp=True)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    trainable_vars=tf.trainable_variables()

    variables_to_restore = [v for v in trainable_vars if v.name.find("mp")==-1 ]
    mp_vars = [v for v in trainable_vars if v.name.find("mp") != -1]
    layer_norm2 = [v for v in trainable_vars if v.name.find("layer_norm2") != -1]# we suggest retraining layer_norm2 for parallel insertion--sometimes slightly better, around 0.3-0.5%

    sess.run(init)

    saver = tf.train.Saver(variables_to_restore)
    saver.restore(sess, "Data/Models/generation_model/model_nextitnet_transfer_pretrain.ckpt")
    print sess.run(variables_to_restore[0])

    # source_item_embedding=tf.reduce_mean(itemrec.dilate_input,1)
    source_item_embedding = tf.reduce_mean(itemrec.dilate_input[:,-1:,:], 1)# use the last token
    # source_item_embedding=tf.add(itemrec.dilate_input[:,0,:],itemrec.dilate_input[:,-2,:])
    embedding_size = tf.shape(source_item_embedding)[1]
    with tf.variable_scope("target-item"):
        allitem_embeddings_target = tf.get_variable('allitem_embeddings_target',
                                                    [model_para['target_item_size'],
                                                     model_para['dilated_channels']],
                                                    initializer=tf.truncated_normal_initializer(stddev=0.02),
                                                    regularizer=tf.contrib.layers.l2_regularizer(0.02)
                                                    )
        is_training = tf.placeholder(tf.bool, shape=())

        # training
        itemseq_input_target_pos = tf.placeholder('int32',
                                                  [None, None], name='itemseq_input_pos')
        itemseq_input_target_neg = tf.placeholder('int32',
                                                  [None, None], name='itemseq_input_neg')
        target_item_embedding_pos = tf.nn.embedding_lookup(allitem_embeddings_target,
                                                           itemseq_input_target_pos,
                                                           name="target_item_embedding_pos")
        target_item_embedding_neg = tf.nn.embedding_lookup(allitem_embeddings_target,
                                                           itemseq_input_target_neg,
                                                           name="target_item_embedding_neg")

        pos_score = source_item_embedding * tf.reshape(target_item_embedding_pos, [-1, embedding_size])
        neg_score = source_item_embedding * tf.reshape(target_item_embedding_neg, [-1, embedding_size])
        pos_logits = tf.reduce_sum(pos_score, -1)
        neg_logits = tf.reduce_sum(neg_score, -1)


        # testing
        itemseq_input_target_label = tf.placeholder('int32',
                                                    [None, None], name='itemseq_input_target_label')
        tf.add_to_collection("itemseq_input_target_label", itemseq_input_target_label)

        target_label_item_embedding = tf.nn.embedding_lookup(allitem_embeddings_target,
                                                             itemseq_input_target_label,
                                                             name="target_label_item_embedding")

        source_item_embedding_test = tf.expand_dims(source_item_embedding, 1)  # (batch, 1, embeddingsize)
        target_item_embedding = tf.transpose(target_label_item_embedding, [0, 2, 1])  # transpose
        score_test = tf.matmul(source_item_embedding_test, target_item_embedding)
        top_k_test = tf.nn.top_k(score_test[:, :], k=top_k, name='top-k')
        tf.add_to_collection("top_k", top_k_test[1])


        loss = tf.reduce_mean(
            - tf.log(tf.sigmoid(pos_logits) + 1e-24) -
            tf.log(1 - tf.sigmoid(neg_logits) + 1e-24)
        )
        #BPR loss
        # loss=-tf.reduce_mean(tf.log(tf.sigmoid(pos_logits-neg_logits)))+ 1e-24
        reg_losses = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss += reg_losses

    sc_variable2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target-item')


    optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1, name='Adam').minimize(loss,var_list=[sc_variable2,mp_vars,layer_norm2])
    unitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            unitialized_vars.append(var)

    initialize_op = tf.variables_initializer(unitialized_vars)
    # vars = tf.trainable_variables()
    sess.run(initialize_op)



    numIters = 1
    for iter in range(model_para['iterations']):
        batch_no = 0
        batch_size = model_para['batch_size']
        while (batch_no + 1) * batch_size < train_set.shape[0]:

            start = time.time()
            #the first n-1 is source, the last one is target
            #item_batch=[[1,2,3],[4,5,6]]
            item_batch = train_set[batch_no * batch_size: (batch_no + 1) * batch_size, :]

            pos_batch=item_batch[:,-1]#[3 6] used for negative sampling
            # source_batch=item_batch[:,:-1]#
            pos_target=item_batch[:,-1:]#[[3][6]]
            neg_target = np.array([[random_neq(1, targets_len, s)] for s in pos_batch])



            _, loss_out, reg_losses_out = sess.run(
                [optimizer, loss, reg_losses],
                feed_dict={
                    itemrec.itemseq_input: item_batch,
                    itemseq_input_target_pos:pos_target,
                    itemseq_input_target_neg:neg_target
                })
            end = time.time()
            if numIters % args.eval_iter == 0:
                print "-------------------------------------------------------train1"
                print "LOSS: {}\Reg_LOSS: {}\tITER: {}\tBATCH_NO: {}\t STEP:{}\t total_batches:{}".format(
                    loss_out, reg_losses_out,iter, batch_no, numIters, train_set.shape[0] / batch_size)
                print "TIME FOR BATCH", end - start
                print "TIME FOR ITER (mins)", (end - start) * (train_set.shape[0] / batch_size) / 60.0
            batch_no += 1
            if numIters % args.eval_iter == 0:
                batch_no_test = 0
                batch_size_test = batch_size * 1
                hits = []  # 1
                mrrs = []  # ---add 1

                while (batch_no_test + 1) * batch_size_test < valid_set.shape[0]:
                    if (numIters / (args.eval_iter) < 10):
                        if (batch_no_test > 20):
                            break
                    else:
                        if (batch_no_test > 500):
                            break
                    item_batch = valid_set[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
                    pos_batch = item_batch[:, -1]  # [3 6] used for negative sampling
                    # source_batch = item_batch[:, :-1]  #
                    pos_target = item_batch[:, -1:]  # [[3][6]]
                    # randomly choose 99 negative items
                    neg_target = np.array([random_negs(1, targets_len, negtive_samples, s) for s in pos_batch])
                    target=np.array(np.concatenate([neg_target,pos_target],1))
                    [top_k_batch] = sess.run(
                        [top_k_test],
                        feed_dict={
                            itemrec.itemseq_input: item_batch,
                            itemseq_input_target_label: target
                        })
                    #note that  in top_k_batch[1], such as [1 9 4 5 0], we just need to check whether 0 is here, that's fine
                    top_k=np.squeeze(top_k_batch[1]) #remove one dimension since e.g., [[[1,2,4]],[[34,2,4]]]-->[[1,2,4],[34,2,4]]
                    for i in range(top_k.shape[0]):
                        top_k_per_batch=top_k[i]
                        predictmap = {ch: i for i, ch in enumerate(top_k_per_batch)}  # add 2
                        rank = predictmap.get(negtive_samples)  # add 3
                        if rank == None:
                            hits.append(0.0)
                            mrrs.append(0.0)  # add 5
                        else:
                            hits.append(1.0)
                            mrrs.append(1.0 / (rank + 1))  # add 4
                    batch_no_test += 1
                print "-------------------------------------------------------Accuracy"
                if len(hits)!=0:
                    print "Accuracy hit_n:", sum(hits) / float(len(hits)),"MRR_n:", sum(mrrs) / float(len(mrrs))  # 5

            numIters += 1
            # if numIters % args.save_para_every == 0:
            #     save_path = saver.save(sess,
            #                            "Data/Models/generation_model/nextitnet_cloze_transfer_finetune_avg".format(iter, numIters))

if __name__ == '__main__':
    main()
