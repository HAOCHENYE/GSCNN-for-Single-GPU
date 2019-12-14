This repo is forked from https://github.com/nv-tlabs/GSCNN


Add group normalization and dw conv and delete some image Augmention method. You should change variable root in cityscapes.py before training


Training:


python --adam (--gn) (--dw) 


Currenly, mean iou of my training result  is only 0.4084(with group normaliza, because my memory of GPU is only 8G, BN is hard to use)
