import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LeakyReLU
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_dense_batch


# GAT  model
class GATBert(torch.nn.Module):
    def __init__(self, dropout=0.2):
        super(GATBert, self).__init__()
        self.tanh = nn.Tanh()
        self.l_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.proteinsfile = torch.load('kiba_proteins.pt').cuda()
        # graph layers
        self.gat1 = GATConv(78, 78, heads=10, dropout=dropout)
        self.gat2 = GATConv(78 * 10, 128, dropout=dropout)
        self.fc_1 = nn.Linear(128, 128)

        
        self.after_bert_fc_1 = nn.Linear(1280, 512)
        self.ln_after_fc_1 = nn.LayerNorm(512)
        self.after_bert_fc_2 = nn.Linear(512, 128)

         # Parallel Co-attention
        self.W_b = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(128,128)))
        self.W_c = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(64,128)))
        self.W_p = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(64,128)))
        self.w_hc = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(64,1)))
        self.w_hp = nn.Parameter(nn.init.xavier_normal_(torch.Tensor(64,1)))
        
        # combined layers
        self.after_concat_fc1 = nn.Linear(256, 1024)
        self.after_concat_fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, 1)


    def forward(self, data):
        # graph input feed-forward
        x, edge_index, target_id, batch = (data.x).float(), data.edge_index, data.target_id, data.batch
        # adj = to_dense_adj(edge_index = edge_index, batch = batch, max_num_nodes = 45)
        target = torch.index_select(self.proteinsfile, 0, torch.LongTensor(target_id).cuda())

        x = F.leaky_relu(self.gat1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.gat2(x, edge_index)
        x = self.l_relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        x, mask = to_dense_batch(x, batch, max_num_nodes = 69)
        x = self.fc_1(x)
        x = self.l_relu(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)

        target = self.after_bert_fc_1(target)
        target = self.ln_after_fc_1(target)
        target = self.l_relu(target)
        target = self.after_bert_fc_2(target)
        target = self.l_relu(target)
        target = self.dropout(target)

        compound, protein = self.parallel_co_attention(x, target)
       
        # concat
        c_p = torch.cat((compound, protein), 1)
        c_p = self.dropout(self.l_relu(self.after_concat_fc1(c_p)))
        c_p = self.dropout(self.l_relu(self.after_concat_fc2(c_p)))
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
