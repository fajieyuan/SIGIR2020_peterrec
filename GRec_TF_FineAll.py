import tensorflow as tf
import data_loader_recsys_transfer_context_ as data_loader_recsys
import utils
import shutil
import time
import math
import eval
import numpy as np
import argparse
import sys
import ops


def shuffleseq(train_set,padtoken):
    shuffle_seqtrain = []
    for i in range(len(train_set)):
        # print x_train[i]
        seq = train_set[i][1:]
        lenseq = len(seq)
        # split=np.split(padtoken)
        copyseq=list(seq)
        padcount = copyseq.count(padtoken)  #the number of padding elements
        copyseq = copyseq[padcount:] # the remaining elements
        # copyseq=seq
        shuffle_indices = np.random.permutation(np.arange(len(copyseq)))
        # list to array
        copyseq= np.array(copyseq)
        copyseq_shuffle=copyseq[shuffle_indices]

        padtoken_list=[padtoken]*padcount
        # array to list, + means concat in list and  real plus in array
        seq=list(train_set[i][0:1])+padtoken_list+list(copyseq_shuffle)
        shuffle_seqtrain.append(seq)
    x_train = np.array(shuffle_seqtrain)  # list to ndarray
    print "shuffling is done!"
    return x_train


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
                        help='Sample from top k predictions, you cannot set top_k=1 due to evaluation np.squeeze')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='hyperpara-Adam')
    parser.add_argument('--eval_iter', type=int, default=2,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=2,
                        help='save model parameters every')
    parser.add_argument('--datapath', type=str, default='Data/Session/history_sequences_20181014_fajie_transfer_finetune_small.csv',
                        help='data path')
    parser.add_argument('--tt_percentage', type=float, default=0.1,
                        help='default=0.2 means 80% training 20% testing')
    parser.add_argument('--is_generatesubsession', type=bool, default=False,
                        help='whether generating a subsessions, e.g., 12345-->01234,00123,00012  It may be useful for very some very long sequences')

    parser.add_argument('--padtoken', type=str, default='0',
                        help='is the padding token in the beggining of the sequence')
    parser.add_argument('--negtive_samples', type=int, default='99',
                        help='the number of negative examples for each positive one')
    parser.add_argument('--is_shuffle', type=bool, default=False,
                        help='whether shuffle the training and testing dataset, e.g., 012345-->051324')
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
    dev_sample_index = -1 * int(args.tt_percentage * float(len(all_samples)))
    train_set, valid_set = all_samples[:dev_sample_index], all_samples[dev_sample_index:]


    model_para = {
        'item_size': len(items),
        'target_item_size': len(targets),
        'dilated_channels': 64,
        'dilations': [1,4,1,4,1,4,1,4,],
        'kernel_size': 3,
        'learning_rate':0.001,
        'batch_size':2, #you can not use batch_size=1 since in the following you use np.squeeze will reuduce one dimension
        'iterations': 400,
        'is_negsample':False #False denotes no negative sampling
    }
    sess = tf.Session()

    new_saver = tf.train.import_meta_graph('Data/Models/generation_model/model_nextitnet_cloze.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('Data/Models/generation_model/'))
    graph = tf.get_default_graph()

    trainable_vars = tf.trainable_variables()
    itemseq_input = graph.get_tensor_by_name("itemseq_input:0")

    allitem_embeddings=tf.trainable_variables()[0]
    dilate_input=tf.get_collection("dilate_input")[0]

    cnn_vars = []
    for i in range(64):
        cnn_vars.append(tf.trainable_variables()[i])
    context_embedding = tf.get_collection("context_embedding")[0]
    print "allitem_embeddings", (sess.run(allitem_embeddings))


    source_item_embedding=tf.reduce_mean(dilate_input,1)
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
        # BPR
        #loss = -tf.reduce_mean(tf.log(tf.sigmoid(pos_logits - neg_logits))) + 1e-24
        reg_losses = tf.reduce_mean(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        loss += reg_losses



    sc_variable2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target-item')
    optimizer = tf.train.AdamOptimizer(model_para['learning_rate'], beta1=args.beta1, name='Adam2').minimize(loss)
    # optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    unitialized_vars = []
    for var in tf.global_variables():
        try:
            sess.run(var)
        except tf.errors.FailedPreconditionError:
            unitialized_vars.append(var)

    initialize_op = tf.variables_initializer(unitialized_vars)
    vars = tf.trainable_variables()
    sess.run(initialize_op)
    saver = tf.train.Saver()

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
            source_batch=item_batch[:,:-1]#
            pos_target=item_batch[:,-1:]#[[3][6]]
            #here  you should use random_neq(1, targets_len-1, s)
            # neg_target=np.array([[random_neq(1, targets_len-1, pos_batch)] for s in pos_batch])
            # not include targets_len
            neg_target = np.array([[random_neq(1, targets_len, s)] for s in pos_batch])
            # neg_target=random_neq(1, len(targets)-1, pos_batch) #remove the first 0-unk


            _, loss_out, reg_losses_out = sess.run(
                [optimizer, loss, reg_losses],
                feed_dict={
                    itemseq_input: source_batch,
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
                # batch_size_test =  1
                hits = []  # 1

                while (batch_no_test + 1) * batch_size_test < valid_set.shape[0]:
                    if (numIters / (args.eval_iter) < 10):
                        if (batch_no_test > 20):
                            break
                    else:
                        if (batch_no_test > 500):
                            break
                    item_batch = valid_set[batch_no_test * batch_size_test: (batch_no_test + 1) * batch_size_test, :]
                    pos_batch = item_batch[:, -1]  # [3 6] used for negative sampling
                    source_batch = item_batch[:, :-1]  #
                    pos_target = item_batch[:, -1:]  # [[3][6]]
                    # randomly choose 999 negative items
                    neg_target = np.array([random_negs(1, targets_len, negtive_samples, s) for s in pos_batch])
                    target=np.array(np.concatenate([neg_target,pos_target],1))


                    [top_k_batch] = sess.run(
                        [top_k_test],
                        feed_dict={
                            itemseq_input: source_batch,
                            itemseq_input_target_label: target
                        })

                    #note that  in top_k_batch[1], such as [1 9 4 5 0], we just need to check whether 0 is here, that's fine
                    top_k=np.squeeze(top_k_batch[1]) #remove one dimension since e.g., [[[1,2,4]],[[34,2,4]]]-->[[1,2,4],[34,2,4]]
                    for i in range(top_k.shape[0]):
                        top_k_per_batch=top_k[i]
                        if negtive_samples in top_k_per_batch: # we just need to check whether index 0 (i.e., the ground truth is in the top-k)
                            hits.append(1.0)
                        else:
                            hits.append(0.0)
                    batch_no_test += 1
                print "-------------------------------------------------------Accuracy"
                if len(hits)!=0:
                    print "Accuracy hit_5:", sum(hits) / float(len(hits))  # 5
            numIters += 1
            if numIters % args.save_para_every == 0:
                save_path = saver.save(sess,
                                       "Data/Models/generation_model/nextitnet_cloze_transfer_finetune_avg".format(iter, numIters))

if __name__ == '__main__':
    main()
