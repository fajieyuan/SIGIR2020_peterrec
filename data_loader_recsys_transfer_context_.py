import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tensorflow.contrib import learn
from collections import Counter

# This Data_Loader file is copied online
#add context such as userID
class Data_Loader:
    def __init__(self, options):

        positive_data_file = options['dir_name']
        positive_examples = list(open(positive_data_file, "r").readlines())
        # positive_examples = [[s[0],s[2:]]for s in positive_examples]

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
        self.target = np.array(list(vocab_processor_target.fit_transform(target))) #pad 0 in the end
        self.target_dict = vocab_processor_target.vocabulary_._mapping

        self.separator = len(self.item)+len(self.target)# it is just used for separating such as :
        lens=self.item.shape[0]
        # sep=np.full((lens, 1), self.separator)


        # self.example = np.hstack((self.item,sep,self.target))
        # concat source and one target

        self.example=np.array([])
        self.example=[]
        for line in range(lens):
            source_line=  self.item [line]
            target_line=  self.target[line]
            target_nume=len(target_line)
            for j in range(target_nume):
                if target_line[j] !=0:
                    unit = np.append(source_line, np.array(target_line[j]))
                    self.example.append(unit)

        self.example=np.array(self.example)



