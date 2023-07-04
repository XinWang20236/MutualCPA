import numpy as np
import pandas as pd
import sys, os
from random import shuffle
from sqlalchemy import true
import torch
import torch.nn as nn
from model.GCN_Bert import GCNBert
from model.GAT_Bert import GATBert
from model.GIN_Bert import GINBert
from torch.optim import AdamW
from utils1 import *
from torch.utils.tensorboard import SummaryWriter

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    train_loss = 0
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}'.format(epoch,
                                                                           batch_idx * len(data.target_id),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))
    train_loss /= len(train_loader)
    print('Train epoch: {} \tLoss: {:.4f}'.format(epoch, train_loss))
    return train_loss

def predicting(model, device, loader, epoch):
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    model.eval()
    test_loss = 0
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
            test_loss += loss.item()
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    test_loss /= len(loader)
    print('Test\Valid epoch:{}\tLoss:{:.4f}'.format(epoch, test_loss))
    return test_loss, total_labels.numpy().flatten(), total_preds.numpy().flatten()


datasets = [['davis'][int(sys.argv[1])]] #, 'kiba'
modeling = [GCNBert, GINBert, GATBert][int(sys.argv[2])]
model_st = modeling.__name__


TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LOG_INTERVAL = 20
LR = 0.0002
NUM_EPOCHS = 2500
writer = SummaryWriter()


print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)


# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)
    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    train_data = CPADataset(root='data', dataset=dataset+'_train')
    test_data = CPADataset(root='data', dataset=dataset+'_test')
    # train_data = Load_Processed_Dataset(processed_data_file_train)
    # test_data = Load_Processed_Dataset(processed_data_file_test)

    # make data PyTorch mini-batch processing ready
    train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=8, pin_memory = true)

    # training the model
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)
    loss_fn = nn.MSELoss()
    weight_params = [param for name, param in model.named_parameters() if "bias" not in name and "bn" not in name and "norm" not in name]#and ("bn" not in name) and ("norm" not in name)
    bias_params = [param for name, param in model.named_parameters() if "bias" in name]
    optimizer = AdamW([{'params': weight_params, 'weight_decay':1e-4}, 
                       {'params': bias_params, 'weight_decay':0}], lr=LR)
    best_mse = 1000
    best_ci = 0
    best_epoch = -1
    
    model_file_name = 'model_' + model_st + '_' + dataset + '.model'
    result_file_name = 'result_' + model_st + '_' + dataset + '.csv'
    

    for epoch in range(NUM_EPOCHS):
        train_loss = train(model, device, train_loader, optimizer, epoch + 1)
        writer.add_scalar('Loss/Train', train_loss, epoch+1)
        
        test_loss, G, P = predicting(model, device, test_loader, epoch + 1)
        ret = [0, 0, 0, get_rm2(G, P), 0]
        writer.add_scalar('Loss/Test', test_loss, epoch+1)

        if test_loss < best_mse:
            torch.save(model.state_dict(), model_file_name)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            best_epoch = epoch + 1
            best_mse = test_loss
            # best_ci = ret[-1]
            best_rm_2 = ret[3]
            print('mse improved at epoch ', best_epoch, '; best_mse,best_rm_2:', best_mse, best_rm_2, model_st, dataset)
        else:
            print(ret[1], 'No improvement since epoch ', best_epoch, '; best_mse,best_rm_2:', best_mse, best_rm_2, model_st, dataset)

    # for epoch in range(NUM_EPOCHS):
    #     train_loss = train(model, device, train_loader, optimizer, epoch + 1)
    #     writer.add_scalar('Loss/Train', train_loss, epoch+1)
        
    #     test_loss, G, P = predicting(model, device, test_loader, epoch + 1)
    #     ret = [rmse(G, P), mse(G, P), 0, 0, ci(G, P)]
    #     writer.add_scalar('Loss/Test', test_loss, epoch+1)
        
    #     if ret[1] < best_mse:
    #         torch.save(model.state_dict(), model_file_name)
    #         with open(result_file_name, 'w') as f:
    #             f.write(','.join(map(str, ret)))
    #         best_epoch = epoch + 1
    #         best_mse = ret[1]
    #         best_ci = ret[-1]
    #         print('rmse improved at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, model_st, dataset)
    #     else:
    #         print(ret[1], 'No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci, model_st, dataset)