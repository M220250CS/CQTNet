######---------------------------------------------------------------------------------------------------
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
import torch
from cqt_loader import *
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import models
from config import opt
from torchnet import meter
from tqdm import tqdm
import numpy as np
import time
import torch.nn.functional as F
import torch
import torch.nn as nn
from utility import *
from models.CQTNet import CQTNet

# multi_size train
def multi_train(**kwargs):
    parallel = True 
    opt.model = 'CQTNet'
    opt.notes='CQTNet'
    opt.batch_size=32
    opt._parse(kwargs)
    
    # step1: configure model
    model = CQTNet() 
    if parallel: 
        model = torch.nn.DataParallel(model)
    if opt.load_latest:
        model.module.load_latest(opt.notes) if parallel else model.load_latest(opt.notes)
    elif opt.load_model_path:
        model.module.load(opt.load_model_path) if parallel else model.load(opt.load_model_path)
    model.to(opt.device)
    print(model)
    
    # step2: data
    train_data0 = CQT('train', out_length=200)
    train_data1 = CQT('train', out_length=300)
    train_data2 = CQT('train', out_length=400)
    val_data80 = CQT('songs80', out_length=None)
    val_data = CQT('val', out_length=None)
    test_data = CQT('test', out_length=None)
    
    train_dataloader0 = DataLoader(train_data0, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate_fn)
    train_dataloader1 = DataLoader(train_data1, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate_fn)
    train_dataloader2 = DataLoader(train_data2, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate_fn)
    val_dataloader = DataLoader(val_data, 1, shuffle=False, num_workers=1, collate_fn=custom_collate_fn)
    test_dataloader = DataLoader(test_data, 1, shuffle=False, num_workers=1, collate_fn=custom_collate_fn)
    val_dataloader80 = DataLoader(val_data80, 1, shuffle=False, num_workers=1, collate_fn=custom_collate_fn)
    
    # step3: criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.module.parameters() if parallel else model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt.lr_decay, patience=2, verbose=True, min_lr=5e-6)
    
    # train
    best_MAP = 0
    val_slow(model, val_dataloader80, -1)
    for epoch in range(opt.max_epoch):
        running_loss = 0
        num = 0
        for (data0, label0), (data1, label1), (data2, label2) in tqdm(zip(train_dataloader0, train_dataloader1, train_dataloader2)):
            for flag in range(3):
                if flag == 0:
                    data, label = data0, label0
                elif flag == 1:
                    data, label = data1, label1
                else:
                    data, label = data2, label2
                
                input = data.requires_grad_().to(opt.device)
                target = label.to(opt.device)

                optimizer.zero_grad()
                score, _ = model(input)
                loss = criterion(score, target)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                num += target.shape[0]
        
        running_loss /= num 
        print(running_loss)
        if parallel:
            model.module.save(opt.notes)
        else:
            model.save(opt.notes)
        
        scheduler.step(running_loss) 
        
        # validate
        MAP = 0
        MAP += val_slow(model, val_dataloader80, epoch)
        if MAP > best_MAP:
            best_MAP = MAP
            print('*****************BEST*****************')
        print('')
        model.train()

@torch.no_grad()
def val_slow(model, dataloader, epoch):
    model.eval()
    labels, features = None, None
    for data, label in dataloader:
        input = data.to(opt.device)
        score, feature = model(input)
        feature = feature.data.cpu().numpy()
        label = label.data.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, feature), axis=0)
            labels = np.concatenate((labels, label))
        else:
            features = feature
            labels = label
    
    features = norm(features)
    dis2d = -np.matmul(features, features.T)
    
    if len(labels) == 80:
        MAP, top10, rank1 = calc_MAP(dis2d, labels, [80, 160])
    else:
        MAP, top10, rank1 = calc_MAP(dis2d, labels)
    
    print(epoch, MAP, top10, rank1)
    model.train()
    return MAP

def test(**kwargs):
    opt.batch_size = 1
    opt.num_workers = 1
    opt.model = 'CQTNet'
    opt.load_latest = False
    opt.load_model_path = 'check_points/CQTNet.pth'
    opt._parse(kwargs)
    
    model = CQTNet()
    if opt.load_latest:
        model.load_latest(opt.notes)
    elif opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    val_data80 = CQT('songs80', out_length=None)
    val_data = CQT('val', out_length=None)
    test_data = CQT('test', out_length=None)
    
    val_dataloader = DataLoader(val_data, 1, shuffle=False, num_workers=1)
    test_dataloader = DataLoader(test_data, 1, shuffle=False, num_workers=1)
    val_dataloader80 = DataLoader(val_data80, 1, shuffle=False, num_workers=1)
    
    val_slow(model, val_dataloader80, 0)
    val_slow(model, val_dataloader, 0)

if __name__ == '__main__':
    import fire
    fire.Fire({
        'multi_train': multi_train,
        'test': test
    })




# ######---------------------------------------------------------------------------------------------------
# import os
# import torch
# from cqt_loader import *
# from torch.utils.data import DataLoader
# import torchvision.transforms as transforms
# import models
# from config import opt
# from torchnet import meter
# from tqdm import tqdm
# import numpy as np
# import time
# import torch.nn.functional as F
# import torch
# import torch.nn as nn
# from utility import *
# from models.CQTNet import CQTNet

# # multi_size train
# def multi_train(**kwargs):
#     parallel = True 
#     opt.model = 'CQTNet'
#     opt.notes='CQTNet'
#     opt.batch_size=32
#     opt._parse(kwargs)
    
#     # step1: configure model
#     model = CQTNet() 
#     if parallel: 
#         model = torch.nn.DataParallel(model)
#     if opt.load_latest:
#         model.module.load_latest(opt.notes) if parallel else model.load_latest(opt.notes)
#     elif opt.load_model_path:
#         model.module.load(opt.load_model_path) if parallel else model.load(opt.load_model_path)
#     model.to(opt.device)
#     print(model)
    
#     # step2: data
#     train_data0 = CQT('train', out_length=200)
#     train_data1 = CQT('train', out_length=300)
#     train_data2 = CQT('train', out_length=400)
#     val_data80 = CQT('songs80', out_length=None)
#     val_data = CQT('val', out_length=None)
#     test_data = CQT('test', out_length=None)
    
#     train_dataloader0 = DataLoader(train_data0, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate_fn)
#     train_dataloader1 = DataLoader(train_data1, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate_fn)
#     train_dataloader2 = DataLoader(train_data2, opt.batch_size, shuffle=True, num_workers=opt.num_workers, collate_fn=custom_collate_fn)
#     val_dataloader = DataLoader(val_data, 1, shuffle=False, num_workers=1, collate_fn=custom_collate_fn)
#     test_dataloader = DataLoader(test_data, 1, shuffle=False, num_workers=1, collate_fn=custom_collate_fn)
#     val_dataloader80 = DataLoader(val_data80, 1, shuffle=False, num_workers=1, collate_fn=custom_collate_fn)
    
#     # step3: criterion and optimizer
#     criterion = torch.nn.CrossEntropyLoss()
#     optimizer = torch.optim.Adam(model.module.parameters() if parallel else model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=opt.lr_decay, patience=2, verbose=True, min_lr=5e-6)
    
#     # train
#     best_MAP = 0
#     val_slow(model, val_dataloader80, -1)
#     for epoch in range(opt.max_epoch):
#         running_loss = 0
#         num = 0
#         for (data0, label0), (data1, label1), (data2, label2) in tqdm(zip(train_dataloader0, train_dataloader1, train_dataloader2)):
#             for flag in range(3):
#                 if flag == 0:
#                     data, label = data0, label0
#                 elif flag == 1:
#                     data, label = data1, label1
#                 else:
#                     data, label = data2, label2
                
#                 input = data.requires_grad_().to(opt.device)
#                 target = label.to(opt.device)

#                 optimizer.zero_grad()
#                 score, _ = model(input)
#                 loss = criterion(score, target)
#                 loss.backward()
#                 optimizer.step()

#                 running_loss += loss.item()
#                 num += target.shape[0]
        
#         running_loss /= num 
#         print(running_loss)
#         if parallel:
#             model.module.save(opt.notes)
#         else:
#             model.save(opt.notes)
        
#         scheduler.step(running_loss) 
        
#         # validate
#         MAP = 0
#         MAP += val_slow(model, val_dataloader80, epoch)
#         if MAP > best_MAP:
#             best_MAP = MAP
#             print('*****************BEST*****************')
#         print('')
#         model.train()

# @torch.no_grad()
# def val_slow(model, dataloader, epoch):
#     model.eval()
#     labels, features = None, None
#     for data, label in dataloader:
#         input = data.to(opt.device)
#         score, feature = model(input)
#         feature = feature.data.cpu().numpy()
#         label = label.data.cpu().numpy()
#         if features is not None:
#             features = np.concatenate((features, feature), axis=0)
#             labels = np.concatenate((labels, label))
#         else:
#             features = feature
#             labels = label
    
#     features = norm(features)
#     dis2d = -np.matmul(features, features.T)
    
#     if len(labels) == 80:
#         MAP, top10, rank1 = calc_MAP(dis2d, labels, [80, 160])
#     else:
#         MAP, top10, rank1 = calc_MAP(dis2d, labels)
    
#     print(epoch, MAP, top10, rank1)
#     model.train()
#     return MAP

# def test(**kwargs):
#     opt.batch_size = 1
#     opt.num_workers = 1
#     opt.model = 'CQTNet'
#     opt.load_latest = False
#     opt.load_model_path = 'check_points/CQTNet.pth'
#     opt._parse(kwargs)
    
#     model = CQTNet()
#     if opt.load_latest:
#         model.load_latest(opt.notes)
#     elif opt.load_model_path:
#         model.load(opt.load_model_path)
#     model.to(opt.device)

#     val_data80 = CQT('songs80', out_length=None)
#     val_data = CQT('val', out_length=None)
#     test_data = CQT('test', out_length=None)
    
#     val_dataloader = DataLoader(val_data, 1, shuffle=False, num_workers=1)
#     test_dataloader = DataLoader(test_data, 1, shuffle=False, num_workers=1)
#     val_dataloader80 = DataLoader(val_data80, 1, shuffle=False, num_workers=1)
    
#     val_slow(model, val_dataloader80, 0)
#     val_slow(model, val_dataloader, 0)

# if __name__ == '__main__':
#     import fire
#     fire.Fire()
