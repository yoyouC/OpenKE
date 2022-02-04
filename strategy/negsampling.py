import torch
import torch.nn as nn

class NegativeSampling(nn.Module):

    def __init__(self, model, dataset, device):
        super(NegativeSampling, self).__init__()
        self.model = model
        self.device = device
        self.dataset = dataset

    def forward(self, triplets):
        hs, rs, ts = triplets
        head_or_tail = torch.randint(high=2, size=hs.size())
        random_entities = torch.randint(high=self.dataset.get_tot_ent(), size=hs.size())
        broken_heads = torch.where(head_or_tail == 1, random_entities, hs)
        broken_tails = torch.where(head_or_tail == 0, random_entities, ts)

        pos_triplets = torch.stack((hs, rs, ts), dim=1)
        neg_triplets = torch.stack((broken_heads, rs, broken_tails), dim=1)
        pos_triplets.to(self.device)
        neg_triplets.to(self.device)

        return self.model(pos_triplets, neg_triplets)
