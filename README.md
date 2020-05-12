# SIGIR2020_PeterRec
# Parameter-Efficient Transfer from Sequential Behaviors for User Modeling and Recommendation
Chinese Introduction: https://zhuanlan.zhihu.com/p/139048117

<p align="center">
    <br>
    <img src="https://pic1.zhimg.com/80/v2-68007f2a6b6fd29807d76065137a9cd8_1440w.jpg" width="400"/>
    <br>
<p>
  
```
Please cite our paper if you use our code or datasets in your publication.
@article{yuan2020parameter,
  title={Parameter-Efficient Transfer from Sequential Behaviors for User Modeling and Recommendation},
  author={Yuan, Fajie and He, Xiangnan and Karatzoglou, Alexandros and Zhang, Liguang},
  journal={arXiv preprint arXiv:2001.04253},
  year={2020}
}
```


PeterRec_cau_parallel.py: PeterRec with causal cnn and parallel insertion

PeterRec_cau_serial.py: PeterRec with causal cnn and serial insertion

PeterRec_noncau_parallel.py: PeterRec with noncausal cnn and parallel insertion

PeterRec_noncau_serial.py: PeterRec with causal cnn and serial insertion



NextitNet_TF_Pretrain.py: Petrained by NextItNet [0] (i.e., causal cnn)

GRec_TF_Pretrain.py: Petrained by the encoder of GRec [1] (i.e., noncausal cnn)


## Demo Steps:

You can directly run our code:

First:  python NextitNet_TF_Pretrain.py

After convergence(you can stop it once the pretrained model is saved!)

Second: python PeterRec_cau_serial.py

Make sure PeterRec is converged, sometimes it will keep relatively stable for several iteractions and then keep increasing again.

or

First:  python GRec_TF_Pretrain.py

Second: python PeterRec_noncau_parallel.py

## Running our paper:
Replacing the demo dataset with our public datasets (including both pretraining and finetuning):

You will reproduce the results reported in our paper using our papar settings, including learning rate, embedding size,
dilations, batch size, etc. Note that the results reported in the paper are based on the same hyper-parameter settings for fair comparison and ablation tests. You may further finetune hyper-parameters to obtatin the best performance. For example, we use 0.001 as learning rate during finetuning, you may find 0.0001 performs better although all insights in the paper keep consistent.
In addition, there are some other improvement places, such as the negative sampling used for funetuning. For simplicity, we implement a very basic one by uniform sampling, suggest you using more advanced sampler such as LambdaFM 
(LambdaFM: Learning Optimal Ranking with Factorization Machines Using Lambda Surrogates). 
### DataSet （desensitized ）Link
```
ColdRec2: https://drive.google.com/open?id=1OcvbBJN0jlPTEjE0lvcDfXRkzOjepMXH
ColdRec1: https://drive.google.com/open?id=1N7pMXLh8LkSYDX30-zId1pEMeNDmA7t6
```

### recommendation settings
NextitNet_TF_Pretrain.py

    parser.add_argument('--eval_iter', type=int, default=10000,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=10000,
                        help='save model parameters every')
    parser.add_argument('--datapath', type=str, default='Data/Session/coldrec2_pre.csv',
                        help='data path')
    model_para = {
        'item_size': len(items),
        'dilated_channels': 256,
        'dilations': [1,4,1,4,1,4,1,4,],
        'kernel_size': 3,
        'learning_rate':0.001,
        'batch_size':16,# you can try 32, 64, 128, 256, etc.
        'iterations':100, #you can stop it once converged
        'is_negsample':True #False denotes no negative sampling
    }
     
                        
PeterRec settings (E.g.,PeterRec_cau_serial.py):

    parser.add_argument('--eval_iter', type=int, default=500,
                        help='Sample generator output evry x steps')
    parser.add_argument('--save_para_every', type=int, default=500,
                        help='save model parameters every')
    parser.add_argument('--datapath', type=str, default='Data/Session/coldrec2_fine.csv',
                        help='data path')
    model_para = {
        'item_size': len(items),
        'target_item_size': len(targets),
        'dilated_channels': 256,
        'cardinality': 1, # 1 is ResNet, otherwise is ResNeXt (performs similarly, but slowly)
        'dilations': [1,4,1,4,1,4,1,4,],
        'kernel_size': 3,
        'learning_rate':0.0001,
        'batch_size':512, #you can not use batch_size=1 since in the following you use np.squeeze will reuduce one dimension
        'iterations': 100,
        'has_positionalembedding': args.has_positionalembedding
    }
  

## Environments
* Tensorflow (version: 1.7.0)
* python 2.7

## Related work:
```
[1]
@inproceedings{yuan2019simple,
  title={A simple convolutional generative network for next item recommendation},
  author={Yuan, Fajie and Karatzoglou, Alexandros and Arapakis, Ioannis and Jose, Joemon M and He, Xiangnan},
  booktitle={Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining},
  pages={582--590},
  year={2019}
}
```
```
[2]
@inproceedings{yuan2020future,
  title={Future Data Helps Training: Modeling Future Contexts for Session-based Recommendation},
  author={Yuan, Fajie and He, Xiangnan and Jiang, Haochuan and Guo, Guibing and Xiong, Jian and Xu, Zhezhao and Xiong, Yilin},
  booktitle={Proceedings of The Web Conference 2020},
  pages={303--313},
  year={2020}
}
```
```
[3]
@article{sun2020generic,
  title={A Generic Network Compression Framework for Sequential Recommender Systems},
  author={Sun, Yang and Yuan, Fajie and Yang, Ming and Wei, Guoao and Zhao, Zhou and Liu, Duo},
  journal={arXiv preprint arXiv:2004.13139},
  year={2020}
}
```
