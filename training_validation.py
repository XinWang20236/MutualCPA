import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from model.GCN_Bert import GCNBert
from model.GAT_Bert import GATBert
from model.GIN_Bert import GINBert
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import *
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
        
        # torch.cuda.empty_cache()
    train_loss /= len(train_loader)
    print('Train epoch:{}\tLoss:{:.4f}'.format(epoch, train_loss))
    return train_loss
    


def predicting(model, device, loader, epoch):
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    val_loss = 0
    model.eval()
    total_preds = torch.Tensor().cuda()
    total_labels = torch.Tensor().cuda() 
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
            val_loss += loss.item()
            total_preds = torch.cat((total_preds, output), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1)), 0)
            # torch.cuda.empty_cache()
    val_loss /= len(loader)
    print('Test\Valid epoch:{}\tLoss:{:.4f}'.format(epoch, val_loss))
    return val_loss, total_labels.cpu().numpy().flatten(), total_preds.cpu().numpy().flatten()


datasets = [['kiba'][int(sys.argv[1])]] #, 'kiba''davis'
modeling = [GCNBert, GINBert, GATBert][int(sys.argv[2])]
model_st = modeling.__name__


TRAIN_BATCH_SIZE = 512
TEST_BATCH_SIZE = 512
LOG_INTERVAL = 40
LR = 0.0001
NUM_EPOCHS = 1500
writer = SummaryWriter()

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)

# Main program: iterate over different datasets
for dataset in datasets:
    print('\nrunning on ', model_st + '_' + dataset)
    processed_data_file_train = 'data/processed/' + dataset + '_train'
    processed_data_file_test = 'data/processed/' + dataset + '_test'
    
    train_data = Load_Processed_Dataset(processed_data_file_train)
    test_data = Load_Processed_Dataset(processed_data_file_test)

    train_size = int(0.8 * len(train_data))
    valid_size = len(train_data) - train_size
    train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

    # make data PyTorch mini-batch processing ready
    
    
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=8)

    # training the model
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    model = modeling()
    # model.load_state_dict(torch.load('./model_GINBert_kiba.model'))
    model.to(device)
    
    loss_fn = nn.MSELoss()
    weight_params = [param for name, param in model.named_parameters() if "bias" not in name]
    bias_params = [param for name, param in model.named_parameters() if "bias" in name]
    optimizer = Adam([{'params': weight_params, 'weight_decay':1e-5}, 
                      {'params': bias_params, 'weight_decay':0}], lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, mode='min',factor=0.9, patience=30)
    best_mse = 1000
    best_test_mse = 1000
    best_test_ci = 0
    best_epoch = -1
        
    model_file_name = 'model_' + model_st + '_' + dataset + '.model'
    result_file_name = 'result_' + model_st + '_' + dataset + '.csv'
        
    for epoch in range(NUM_EPOCHS):
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8)
        train_loss = train(model, device, train_loader, optimizer, epoch + 1)
        
        writer.add_scalar('Loss/Train', train_loss, epoch+1) 
        # train_mse = mse(G, P)
        # writer.add_scalar('MSE/Train', train_mse, epoch+1)
        # train_CI = ci(G, P)
        # writer.add_scalar('CI/Train', train_CI, epoch+1)
        
        print('predicting for valid data')
        valid_loader = DataLoader(valid_data, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=8)
        val_loss, G, P = predicting(model, device, valid_loader, epoch+1)
        scheduler.step(val_loss)
        writer.add_scalar('Loss/Validation', val_loss, epoch+1)
        val_mse = mse(G, P)
        writer.add_scalar('MSE/Validation', val_mse, epoch+1)
        # val_CI = ci(G, P)
        # writer.add_scalar('CI/Validation', val_CI, epoch+1)
        
        if val_mse < best_mse:
            best_mse = val_mse
            best_epoch = epoch + 1
            torch.save(model.state_dict(), model_file_name)
                
            print('predicting for test data')
            test_loss, G, P = predicting(model, device, test_loader, epoch+1)
            ret = [rmse(G, P), mse(G, P), 0, 0, ci(G, P)]
            # writer.add_scalar('Loss/Test', test_loss, epoch+1)
            # writer.add_scalar('MSE/Test', ret[1], epoch+1)
            # writer.add_scalar('CI/Test', ret[-1], epoch+1)
            with open(result_file_name, 'w') as f:
                f.write(','.join(map(str, ret)))
            best_test_mse = ret[1]
            best_test_ci = ret[-1]
            print('rmse improved at epoch ', best_epoch, '; best_test_mse,best_test_ci:', best_test_mse,
                      best_test_ci, model_st, dataset)
            print('\n')
        else:
            print(ret[1], 'No improvement since epoch ', best_epoch, '; best_test_mse,best_test_ci:', best_test_mse,
                      best_test_ci, model_st, dataset)
            print('\n')