import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Sequential, Linear, ReLU, LeakyReLU
from torch_geometric.nn import GINConv
# from torch_geometric.nn.dense import DenseGINConv
from torch_geometric.utils import to_dense_batch

# GINConv model
class GINBert(torch.nn.Module):
    def __init__(self, num_features_xd = 78):

        super(GINBert, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(0.2)
        self.tanh = nn.Tanh()
        self.l_relu = nn.LeakyReLU()
        self.proteinsfile = torch.load('kiba_proteins.pt').cuda()
        # self.proteinsfile = torch.load('davis_proteins.pt').cuda()
        
        # graph convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), LeakyReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), LeakyReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = nn.BatchNorm1d(dim)

        self.fc_1 = nn.Linear(32, 128)
        
        self.after_bert_fc_1 = nn.Linear(1280, 128)
        self.ln_after_fc_1 = nn.LayerNorm(128)
        # self.after_bert_fc_2 = nn.Linear(512, 128)
        
        # Parallel Co-attention
        self.W_b  = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(128,128)))
        self.W_c  = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32,128)))
        self.W_p  = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32,128)))
        self.w_hc = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32,1)))
        self.w_hp = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32,1)))


        # combined layers
        self.after_concat_fc1 = nn.Linear(256, 1024)
        self.after_concat_fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                nn.init.constant_(m.bias, 0)
            
        self.output_c = []
        self.output_p = []

    def forward(self, data):
        x, edge_index, target_id, batch = (data.x).float(), data.edge_index, data.target_id, data.batch
        target = torch.index_select(self.proteinsfile, 0, torch.LongTensor(target_id).cuda())
               
        
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.leaky_relu(self.conv3(x, edge_index))
        x = self.bn3(x)      
        x = F.leaky_relu(self.conv4(x, edge_index))
        x = self.bn4(x)      
        x = F.leaky_relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        
        x, mask = to_dense_batch(x, batch, max_num_nodes = 69)
        
        x = self.l_relu(self.fc_1(x))
        x = self.dropout(x)

        x = x.permute(0, 2, 1)
        
        target = self.after_bert_fc_1(target)
        target = self.ln_after_fc_1(target)
        target = self.l_relu(target)
        target = self.dropout(target)

        
        compound, protein, t_c, t_p = self.parallel_co_attention(x, target)
             
        self.output_c.append(t_c)
        self.output_p.append(t_p)
        
        c_p = torch.cat((compound, protein), -1)
        c_p = self.dropout(self.l_relu(self.after_concat_fc1(c_p)))
        c_p = self.dropout(self.l_relu(self.after_concat_fc2(c_p)))
        out = self.out(c_p)
        
        return out, self.output_c, self.output_p
    
    
    def parallel_co_attention(self, x, target):  # compound : B x 128 x 45, target : B x L x 128
        
        C = self.tanh(torch.matmul(target, torch.matmul(self.W_b, x))) # B x L x 45

        H_c = self.tanh(torch.matmul(self.W_c, x) + torch.matmul(torch.matmul(self.W_p, target.permute(0, 2, 1)), C))          # B x k x 45
        H_p = self.tanh(torch.matmul(self.W_p, target.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_c, x), C.permute(0, 2, 1)))# B x k x L

        a_c = F.softmax(torch.matmul(torch.t(self.w_hc), H_c), dim=2) # B x 1 x 45
        a_p = F.softmax(torch.matmul(torch.t(self.w_hp), H_p), dim=2) # B x 1 x L

        c = torch.squeeze(torch.matmul(a_c, x.permute(0, 2, 1)))      # B x 128
        p = torch.squeeze(torch.matmul(a_p, target))                  # B x 128

        return c, p, a_c, a_p