import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_batch


class GCNBert(torch.nn.Module):
    def __init__(self):
        super(GCNBert, self).__init__()
        
        self.compound_atom_features = 512
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.proteinsfile = torch.load('kiba_proteins.pt').cuda()
        
        # compound g_conv
        self.G_conv1 = GCNConv(78, 128)
        self.G_conv2 = GCNConv(128, 128) 
        self.G_conv3 = GCNConv(128, 256)
        # compound fc after co-attention
        self.fc1 = torch.nn.Linear(256, 1024)
        self.fc2 = torch.nn.Linear(1024, 128)
        
        # Parallel Co-attention
        self.W_b = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(128,128)))
        self.W_c = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32,128)))
        self.W_p = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32,128)))
        self.w_hc = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32,1)))
        self.w_hp = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(32,1)))
        

        
        # protein fc after co-attention
        self.after_bert_fc_1 = nn.Linear(1280, 128)
        self.ln_after_fc_1 = nn.LayerNorm(128)

        # self.after_bert_fc_2 = nn.Linear(512, 128)

        # concat fc
        self.after_concat_fc1 = nn.Linear(256, 1024)
        self.after_concat_fc2 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

    def forward(self, data):
        x, edge_index, target_id, batch = (data.x).float(), data.edge_index, data.target_id, data.batch
        # adj = to_dense_adj(edge_index = edge_index, batch = batch, max_num_nodes = 45)
        target = torch.index_select(self.proteinsfile, 0, torch.LongTensor(target_id).cuda())

        x = self.G_conv1(x, edge_index)  
        x = self.relu(x)
        x = self.G_conv2(x, edge_index)  
        x = self.relu(x)
        x = self.G_conv3(x, edge_index)  
        x = self.relu(x)                   
        
        x, mask = to_dense_batch(x, batch, max_num_nodes = 45)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        x = x.permute(0, 2, 1)    # B X 512 X 45
        
        target = self.after_bert_fc_1(target)
        target = self.relu(target)
        target = self.dropout(target)
        target = self.after_bert_fc_2(target)
        target = self.relu(target)
        target = self.dropout(target)
        
        
        compound, protein = self.parallel_co_attention(x, target)
        
        
        c_p = torch.cat((compound, protein), 1)
        c_p =self.dropout(self.relu(self.concat_fc1(c_p)))
        c_p =self.dropout(self.relu(self.concat_fc2(c_p)))
        out = self.out(c_p)
        return out
    
    def parallel_co_attention(self, x, target):  # compound : B x 128 x 45, target : B x L x 128
        
        C = self.tanh(torch.matmul(target, torch.matmul(self.W_b, x))) # B x L x 45

        H_c = self.tanh(torch.matmul(self.W_c, x) + torch.matmul(torch.matmul(self.W_p, target.permute(0, 2, 1)), C))          # B x k x 45
        H_p = self.tanh(torch.matmul(self.W_p, target.permute(0, 2, 1)) + torch.matmul(torch.matmul(self.W_c, x), C.permute(0, 2, 1)))# B x k x L

        a_c = F.softmax(torch.matmul(torch.t(self.w_hc), H_c), dim=2) # B x 1 x 45
        a_p = F.softmax(torch.matmul(torch.t(self.w_hp), H_p), dim=2) # B x 1 x L

        c = torch.squeeze(torch.matmul(a_c, x.permute(0, 2, 1)))      # B x 128
        p = torch.squeeze(torch.matmul(a_p, target))                  # B x 128

        return c, p