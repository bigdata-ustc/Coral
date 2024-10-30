import torch
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)

class Net(nn.Module):
    def __init__(self, student_n, exer_n, knowledge_n, peer, peer_per, device):
        super(Net, self).__init__()
        self.knowledge_num = knowledge_n
        self.exer_num = exer_n
        self.stu_num = student_n
        self.stu_dim = 20
        self.prednet_input_len = self.stu_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable
        self.pho = 1
        self.top_k_neighbors = None

        self.peer = peer
        self.peer_per = peer_per

        self.device = device

        super(Net, self).__init__()

        # network structure
        self.student_emb = nn.Embedding(self.stu_num, self.stu_dim * self.knowledge_num)
        self.k_difficulty = nn.Embedding(self.exer_num, self.stu_dim)
        self.e_discrimination = nn.Embedding(self.exer_num, 1)
        self.adj_matrix = torch.zeros([self.knowledge_num, self.stu_num, self.stu_num]).to(self.device)
        self.adj_matrix_ = torch.zeros([self.knowledge_num, self.stu_num, self.stu_num]).to(self.device)

        self.adj_matrix2 = torch.zeros([self.knowledge_num, self.stu_num, self.stu_num]).to(self.device)

        # encoder
        self.all_stu_id = torch.LongTensor(list(range(0, self.stu_num))).to(self.device)
        self.encoder = nn.Linear(self.stu_dim, 2 * self.stu_dim)  # 2 * num_components parameters for mean and variance

        self.theta_encoder = nn.Linear(self.stu_dim, 1)
        self.theta_encoder2 = nn.Linear(self.stu_dim, 1)
        self.gnn_channel = nn.Linear(self.stu_dim, self.stu_dim)
        self.gnn_channel2 = nn.Linear(self.stu_dim, self.stu_dim)

        self.prednet_full1 = nn.Linear(self.knowledge_num, self.knowledge_num)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = nn.Linear(self.knowledge_num, self.knowledge_num)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = nn.Linear(self.knowledge_num, 1)

        # initialization
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, status, stu_id, exer_id, kn_emb, batch_count):
        pass


    def apply_clipper(self):
        clipper = NoneNegClipper()
        self.prednet_full1.apply(clipper)
        self.prednet_full2.apply(clipper)
        self.prednet_full3.apply(clipper)


class NoneNegClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
            w = module.weight.data
            w.add_(torch.relu(-w))  # Clipping in-place
