import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tensorflow.contrib import learn
from collections import Counter

# This Data_Loader file is copied online
class Data_Loader:
    def __init__(self, options):

        positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        positive_examples = [s for s in positive_examples]

        max_document_length = max([len(x.split(",")) for x in positive_examples])
        #max_document_length = max([len(x.split()) for x in positive_examples])  #split by space, one or many, not sensitive
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        self.item = np.array(list(vocab_processor.fit_transform(positive_examples)))
        self.item_dict = vocab_processor.vocabulary_._mapping



        # added to calculate word frequency
        # allitems_hassamewords=list()
        # for line in self.item:
        #     for ele in line:
        #         allitems_hassamewords.append(ele)
        #
        # counts = Counter(allitems_hassamewords)
        # most_com=counts.most_common(10)
        # print allitems_hassamewords




