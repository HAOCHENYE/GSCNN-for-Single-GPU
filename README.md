Add group normalization and dw conv function.
Training:
python --adam (--gn) (--dw) 
Currenly, mean iou of my training result  is only 0.4084(with group normaliza, because my memory of GPU is only 8G, BN is hard to use)
