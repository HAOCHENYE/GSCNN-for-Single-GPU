"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
""" 

import importlib
import torch
import logging

def get_net(args, criterion):
    net = get_model(network=args.arch, num_classes=args.dataset_cls.num_classes,
                    criterion=criterion, trunk=args.trunk,pointwise=args.pointwise, gn=args.gn)
    num_params = sum([param.nelement() for param in net.parameters()])
    logging.info('Model params = {:2.1f}M'.format(num_params / 1000000))

    net = net.cuda()
    net = torch.nn.DataParallel(net)
    return net


def get_model(network, num_classes, criterion, trunk, pointwise, gn):
    
    module = network[:network.rfind('.')]                          #module=network.gscnn
    model = network[network.rfind('.')+1:]                         #model=GSCNN
    mod = importlib.import_module(module)                          #mod : from network import gscnn as mod
    net_func = getattr(mod, model)                                 #net_func = gscnn.GSCNN
    print(type(net_func))
    net = net_func(num_classes=num_classes, trunk=trunk, criterion=criterion, pointwise=pointwise, gn=gn)
    return net


