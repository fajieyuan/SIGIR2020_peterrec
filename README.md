# sigir2020_peterrec
## Parameter-Efficient Transfer from Sequential Behaviors for User Profiling and Recommendation
@article{yuan2020parameter,
  title={Parameter-Efficient Transfer from Sequential Behaviors for User Profiling and Recommendation},
  author={Yuan, Fajie and He, Xiangnan and Karatzoglou, Alexandros and Zhang, Liguang},
  journal={arXiv preprint arXiv:2001.04253},
  year={2020}
}


PeterRec_cau_parallel.py: PeterRec with causal cnn and parallel insertion

PeterRec_cau_serial.py: PeterRec with causal cnn and serial insertion

PeterRec_noncau_parallel.py: PeterRec with noncausal cnn and parallel insertion

PeterRec_noncau_serial.py: PeterRec with causal cnn and serial insertion



NextitNet_TF_Pretrain.py: Petrained by NextItNet [0] (i.e., causal cnn)

GRec_TF_Pretrain.py: Petrained by the encoder of GRec [1] (i.e., noncausal cnn)


## Steps:

First:  python NextitNet_TF_Pretrain.py

After convergence(you can stop it once the pretrained model are saved once!)

Second: python PeterRec_cau_serial.py

Make sure PeterRec is converged, sometimes it will keep relatively stable for several iteractions and then keep increasing again.

or

First:  python GRec_TF_Pretrain.py

After convergence

Second: python PeterRec_noncau_parallel.py


Replace with your own datasets or our public datasets (http://...):



Related work:
[1]
@inproceedings{yuan2019simple,
  title={A simple convolutional generative network for next item recommendation},
  author={Yuan, Fajie and Karatzoglou, Alexandros and Arapakis, Ioannis and Jose, Joemon M and He, Xiangnan},
  booktitle={Proceedings of the Twelfth ACM International Conference on Web Search and Data Mining},
  pages={582--590},
  year={2019}
}

[2]
@article{fajie2019modeling,
	title={Future Data Helps Training: Modeling Future Contexts for Session-based Recommendation},
	author={Yuan, Fajie and He, Xiangnan and Jiang, Haochuan and Guo, Guibing and Xiong, Jian and Xu, Zhezhao and Xiong, Yilin},
	journal={The world wide web conference},
	year={2019}
}
