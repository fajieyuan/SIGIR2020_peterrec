import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tensorflow.contrib import learn
from collections import Counter

# This Data_Loader file is copied online
# data format pretrain 1,2,3,4,5,6
#finetune,
#input   1,2,3,4,5,6,,targetIDs
# output 1,2,3,4,5,6,'CLS',targetIDs  'CLS' denotes classifier
import math

class Data_Loader:
    def __init__(self, options):

        positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        # positive_examples = [[s[0],s[2:]]for s in positive_examples]
        rho = options['lambdafm_rho']
        # [user,itemseq] = [[s[0], s[2:]] for s in positive_examples]
        # print user
        colon=",,"
        source = [s.split(colon)[0] for s in positive_examples]
        target= [s.split(colon)[1] for s in positive_examples]


        max_document_length = max([len(x.split(",")) for x in source])
        # max_document_length = max([len(x.split()) for x in positive_examples])  #split by space, one or many, not sensitive
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        self.item = np.array(list(vocab_processor.fit_transform(source)))
        self.item_dict = vocab_processor.vocabulary_._mapping

        max_document_length_target = max([len(x.split(",")) for x in target])
        vocab_processor_target = learn.preprocessing.VocabularyProcessor(max_document_length_target)
        self.target = np.array(list(vocab_processor_target.fit_transform(target)))  # pad 0 in the end
        self.target_dict = vocab_processor_target.vocabulary_._mapping

        # self.separator = len(self.item) + len(self.target)  # it is just used for separating such as :
        # self.separator = len(self.item_dict) # denote '[CLS]'
        self.separator = 0  # denote '[CLS]'
        lens = self.item.shape[0]
        # sep=np.full((lens, 1), self.separator)

        # self.example = np.hstack((self.item,sep,self.target))
        # concat source and one target
        self.maxtarget = len(self.target_dict)
        self.example = np.array([])
        self.example = []

        itemfreq = {}
        self.itemrank = {}
        self.itemrankprob = {}  # (itemID, probability)
        self.prob = []

        for line in range(lens):
            source_line = self.item[line]
            target_line = self.target[line]
            target_num = len(target_line)


            for j in range(target_num):
                if target_line[j] != 0:
                    # np.array(target_line[j])
                    # unit = np.append(np.array(self.separator),source_line)
                    itemfreq[target_line[j]] = itemfreq.setdefault(target_line[j], 0) + 1
                    unit = np.append(source_line,  np.array(self.separator))
                    unit= np.append(unit,np.array(target_line[j]))
                    self.example.append(unit)

        self.example = np.array(self.example)

        sorted_x = sorted(itemfreq.items(), key=lambda kv: kv[1],
                          reverse=True)  # <type 'list'>: [(1, 3), (2, 2), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1)]
        for index, x in enumerate(sorted_x):
            self.itemrank[x[0]] = index  # self.item_dict has additional 'UNK'
            self.itemrankprob[x[0]] = math.exp(-(index + 1) / (rho * self.maxtarget))
        # print self.itemrank #(itemID, rank)
        self.prob = list(
            self.itemrankprob.values())  # be extremely careful since the index of item should +1 as a  real itemID
        sum_ = np.array(self.prob).sum(axis=0)
        self.prob = self.prob / sum_









