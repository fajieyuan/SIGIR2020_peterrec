#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
import random
import time
def sample_top(a=[], top_k=10):
    idx = np.argsort(a)[::-1]
    idx = idx[:top_k]
    probs = a[idx]
    probs = probs / np.sum(probs)
    choice = np.random.choice(idx, p=probs)
    return choice

# fajie
def sample_top_k(a=[], top_k=10):
    idx = np.argsort(a)[::-1]
    idx = idx[:top_k]
    # probs = a[idx]
    # probs = probs / np.sum(probs)
    # choice = np.random.choice(idx, p=probs)
    return idx

print sample_top_k(np.array([0.02,0.01,0.01,0.16,0.8]),3)



# unit_sequence_lens=3 should be >= 2, otherwise there will be a bug, the default value is 3
# item_batch[1] should be larger than unit_sequence_lens


def regenerate_seq_(itembatch,padtoken='0'):

    unit_sequence_lens = 3#定义单位长度，写死了
    generated_itembatch=[]
    generated_itembatch_padend=[]# 每个sequence最后一个index填写0操作

    for seq_num in range(len(itembatch)): #遍历输入的序列数据

        # the first sequence is from the target user, and the second sequence is from a random user, e.g., itembatch[0]=[1,2,3,4,5]
        #第一个sequence是该用户的sequence，第二个是随机采样了一个大小为5的序列，第三个为随机才养了一个大小为5的序列
        sequence=itembatch[seq_num]
        lens = len(sequence)

        def random_generate(sampled_num):
            gen_sequence = random.sample(sequence, lens)
            return gen_sequence
        if  seq_num==0:
            # lens=50, split into several small seqs
            # if lens<=unit_sequence_lens:
            #     padcount=unit_sequence_lens-lens
            #     gen_sequence=[padtoken]*padcount+sequence
            #     generated_itembatch.append(gen_sequence)

            if lens<=2*unit_sequence_lens:
                gen_seq_num=random_generate(lens)
                for i in xrange(len(gen_seq_num)):
                    generated_itembatch.append([gen_seq_num[i]])#make sure it is a list
            elif lens<=3*unit_sequence_lens:
                # randomly select two
                gen_seq_num = random_generate(lens)
                generated_itembatch.append(gen_seq_num[0:unit_sequence_lens])
                generated_itembatch.append(gen_seq_num[1*unit_sequence_lens:2*unit_sequence_lens])
            elif lens <= 5 * unit_sequence_lens:
                gen_seq_num = random_generate(lens)
                generated_itembatch.append(gen_seq_num[0:unit_sequence_lens])
                generated_itembatch.append(gen_seq_num[1*unit_sequence_lens:2 * unit_sequence_lens])
                generated_itembatch.append(gen_seq_num[2*unit_sequence_lens:3 * unit_sequence_lens])
            elif lens <= 9 * unit_sequence_lens:
                gen_seq_num = random_generate(lens)
                generated_itembatch.append(gen_seq_num[0:unit_sequence_lens])
                generated_itembatch.append(gen_seq_num[1 * unit_sequence_lens:2 * unit_sequence_lens])
                generated_itembatch.append(gen_seq_num[2 * unit_sequence_lens:3 * unit_sequence_lens])
                generated_itembatch.append(gen_seq_num[3 * unit_sequence_lens:4 * unit_sequence_lens])
                generated_itembatch.append(gen_seq_num[4 * unit_sequence_lens:5 * unit_sequence_lens])
            else:
                # >=21 5 group
                gen_seq_num = random_generate(lens)
                generated_itembatch.append(gen_seq_num[0:unit_sequence_lens])
                generated_itembatch.append(gen_seq_num[1 * unit_sequence_lens:2 * unit_sequence_lens])
                generated_itembatch.append(gen_seq_num[2 * unit_sequence_lens:3 * unit_sequence_lens])
                generated_itembatch.append(gen_seq_num[3 * unit_sequence_lens:4 * unit_sequence_lens])
                generated_itembatch.append(gen_seq_num[4 * unit_sequence_lens:5 * unit_sequence_lens])
                generated_itembatch.append(gen_seq_num[5 * unit_sequence_lens:6 * unit_sequence_lens])
        else:
        # the second sequence from a random user
            lens_follow=len(generated_itembatch[0]) #make sure they have the same lenghts with the above process
            if lens>=unit_sequence_lens:
                gen_seq_num =  random.sample(sequence, lens_follow)
                # print gen_seq_num
                generated_itembatch.append(gen_seq_num)
                # gen_seq_num = random.sample(sequence, lens_follow)
                # generated_itembatch.append(gen_seq_num)
    # add padding for the last index
    for seq_num in range(len(generated_itembatch)):
        gen_sequence=generated_itembatch[seq_num]+ [padtoken]
        generated_itembatch_padend.append(gen_sequence)


    # print generated_itembatch_padend
    return generated_itembatch_padend




def regenerate_seq(itembatch,padtoken='0',unit_sequence_lens=10):
    generated_itembatch=[]
    generated_itembatch_padend=[]#add padding after the last index
    for seq_num in range(len(itembatch)):
        # the first sequence is from the target user, and the secodne sequence is from a random user, e.g., itembatch[0]=[1,2,3,4,5]
        sequence=itembatch[seq_num]
        lens = len(sequence)
        if seq_num==0:
            # lens=50, split into several small seqs
            if lens<=unit_sequence_lens:
                padcount=unit_sequence_lens-lens
                #['0', '0', '0', '75UhOolrY1GQe4Byi', '72LLb8QF31GPvOe6K', '73bTHTaSE1GNPmJhx', '767NZGg511GX9WhQn', '72mqDJHEf1GQPwgrA']
                gen_sequence=[padtoken]*padcount+sequence
                generated_itembatch.append(gen_sequence)
            else:
                # if the sequence is larger than unit_sequence_lens=8
                subseq_count = lens / unit_sequence_lens+1
                for gen_seq_num in range(subseq_count):
                    if gen_seq_num>=1:
                        break
                    # if len(sequence)==
                    if gen_seq_num==0:
                        gen_sequence = sequence[
                                       -unit_sequence_lens * (gen_seq_num + 1):]
                        generated_itembatch.append(gen_sequence)
                    else:
                        gen_sequence= sequence[
                                       -unit_sequence_lens * (gen_seq_num + 1):-unit_sequence_lens * (gen_seq_num)]
                        if len(gen_sequence)<unit_sequence_lens:
                            padcount = unit_sequence_lens - len(gen_sequence)
                            gen_sequence = [padtoken] * padcount + gen_sequence
                        else:
                            gen_sequence=gen_sequence
                        generated_itembatch.append(gen_sequence)

                def randome_generate(sampled_num):
                    # if sampled_num>unit_sequence_lens:
                    #     n=unit_sequence_lens
                    # else:
                    #     n = random.randint(1, unit_sequence_lens)
                    padcount = unit_sequence_lens - sampled_num
                    randomsampled_seq = random.sample(sequence, sampled_num)  # no repeated item
                    gen_sequence = [padtoken] * padcount + randomsampled_seq
                    generated_itembatch.append(gen_sequence)


                # do it twice

                randome_generate(random.randint(1, unit_sequence_lens))
                # randome_generate(random.randint(1, unit_sequence_lens))
                randome_generate(1)
                randome_generate(1)


        else:
        # the second sequence from a random user
            if lens<=unit_sequence_lens:
                padcount=unit_sequence_lens-lens
                #['0', '0', '0', '75UhOolrY1GQe4Byi', '72LLb8QF31GPvOe6K', '73bTHTaSE1GNPmJhx', '767NZGg511GX9WhQn', '72mqDJHEf1GQPwgrA']
                gen_sequence=[padtoken]*padcount+sequence
                generated_itembatch.append(gen_sequence)
            else:
                gen_sequence=sequence[
                                       -unit_sequence_lens:]
                generated_itembatch.append(gen_sequence)

    # add padding for the last index
    for seq_num in range(len(generated_itembatch)):
        gen_sequence=generated_itembatch[seq_num]+ [padtoken]
        generated_itembatch_padend.append(gen_sequence)


    # print generated_itembatch_padend
    return generated_itembatch_padend





fajieitem1 = '0'
fajieitem2 = '6XmQrjS7b1GTNZCGB'  # jingwen
fajieitem3 = '72LLb8QF31GPvOe6K'  # zhamazha
fajieitem4 = '6S4kEkrrE1GXxcK0D'  # banlangtuan
fajieitem5 = '6ZHN0z1uS1GVe8KVk'  # dilireba

fajieitem6 = '72HlpiyUv1GOR38a6'  # hugenvyou
fajieitem7 = '75p4rLDZ51GSRAZP4'  # yeliangchangge

fajieitem8 = '6YcOkN0RY1GSOm3UK'  # ouyangnana
fajieitem9 = '75p4rLDZ51GSHmd6y'  # zhangmin
fajieitem10 = '74BX1xeD51GKu5gwW'  # gezhongmeinv
fajieitem11 = '6YBjJ5ujA1GS6wtBd'  # mao he xishuai
fajieitem12 = '767NZGg511GX9WhQn'  # liuluoguo
fajieitem13 = '6YcOkN0RY1GPrAIAf'  # sun yaowei
fajieitem14 = '763nKN9qn1GSoF3VV'  # zhangshaohan yinxingdechibang
fajieitem15 = '73bTHTaSE1GNPmJhx'  # diao sha yu
fajieitem16 = '70fWKDPDV1GPVoqO2'  # quanjiafu music hongchenlaiyalai
fajieitem17 = '757HTMXHO1GQAtp4k'  # dulante kuli doutui
fajieitem18 = '6YtvyJIQT1GNyM4Im'  # juxi
fajieitem19 = '75UhOolrY1GQe4Byi' # dayanglu

fajieitem20 = '72mqDJHEf1GQPwgrA'  # xiaohai chifan
fajieitem21 = '6YcOkN0RY1GN4Crjj'  # xiaohailianquan
fajieitem22 = '72mqDJHEf1GNAQiL1'  # xiaohaiqiche
fajieitem23 = '754pYmxBp1GT89zph'  # zhang junning
fajieitem24 = '71sR4fOgm1GOEadPW'  # mingxing shandong

fajieitem25 = '73ESNtFhJ1GLWrBCh'  # jujingwen
fajieitem26 = '6ZYyJ12oE1GLYoqKY'  # chen   qiaoen
fajieitem27 = '6YcOkN0RY1GTg2wPv'  # pig
fajieitem28 = '74380E8pf1GX0qLLR'  # kaoyu
fajieitem29 = '70hww2B2q1GXq5aFf'  # zhaiyezi
fajieitem30 = '74mC4BObB1GUZS9pV'  # pao mian
fajieitem31 = '73ESNtFhJ1GLWrBCh'  # jujingwen
fajieitem32 = '733S2MonS1GIEeO56'  # suyanmeinv
fajieitem33 = '72oFjdHde1GM8itPn'  # tongliya
fajieitem34 = '71AGPPjav1GS5DjwO'  # dilireba
fajieitem35 = '755InKD3v1GX6NQ9V'  # dianying
fajieitem36 = '6YqFEZwMt1GWMvKnU'  # chehuo
fajieitem37 = '74jvNxTTk1GWEn59J'  # dengchao
fajieitem38 = '6YRJm6AAM1GST45sQ'  # cat and crabs

item_batch=[[ fajieitem2,fajieitem3,fajieitem4,fajieitem20, fajieitem23, fajieitem27,fajieitem28,fajieitem35,
                    fajieitem35, fajieitem28, fajieitem27, fajieitem23, fajieitem20, fajieitem12,fajieitem15, fajieitem3, fajieitem19],
                    [fajieitem12,fajieitem15, fajieitem6,fajieitem3,fajieitem27,fajieitem20, fajieitem24, fajieitem18, fajieitem10,fajieitem20,
                    fajieitem2,fajieitem23,fajieitem24,fajieitem25,fajieitem26,fajieitem28,fajieitem29,fajieitem30,fajieitem3,fajieitem33],
                   ]

# item_batch = [
#     [ fajieitem2,fajieitem3,fajieitem4,fajieitem20, fajieitem23, fajieitem27],
#     [fajieitem21, fajieitem22, fajieitem23, fajieitem24, fajieitem25, fajieitem27, fajieitem28, fajieitem29],
# ]
item_batch=[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]]
# self.input_predict = self.input_predict[:, 0:-1]
a=[1]
a=a[0:-1]
print a
# item_batch=[[1,2,3,4,5,6,7],[1,2,3,4,5]]
# item_batch=[[fajieitem1, fajieitem19, fajieitem3,fajieitem15,fajieitem12,fajieitem20],
#                     [fajieitem12,fajieitem15, fajieitem6,fajieitem3,fajieitem27,fajieitem20],
#                    ]

# start = time.time()
# for i in range(1):
#     regenerate_seq_(item_batch)
# end=time.time()
# print "time",end-start




