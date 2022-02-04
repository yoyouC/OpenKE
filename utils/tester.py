import torch
from .metrics import hit_at_k, mrr

class Tester(object):

    def __init__(self, model, data_loader, use_gpu, entity_count):
        self.model = model
        self.use_gpu = use_gpu
        self.data_loader = data_loader
        self.entity_count = entity_count

        if not self.use_gpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
        
        self.model.to(self.device)
    
    def run(self):
        examples_count = 0.0
        hits_at_1 = 0.0
        hits_at_3 = 0.0
        hits_at_10 = 0.0
        mrr_score = 0.0

        entity_ids = torch.arange(end=self.entity_count, device=self.device).unsqueeze(0)
        for head, relation, tail in self.data_loader:
            current_batch_size = head.size()[0]

            head, relation, tail = head.to(self.device), relation.to(self.device), tail.to(self.device)
            all_entities = entity_ids.repeat(current_batch_size, 1)
            heads = head.reshape(-1, 1).repeat(1, all_entities.size()[1])
            relations = relation.reshape(-1, 1).repeat(1, all_entities.size()[1])
            tails = tail.reshape(-1, 1).repeat(1, all_entities.size()[1])

            # Check all possible tails
            triplets = torch.stack((heads, relations, all_entities), dim=2).reshape(-1, 3)
            data = {
                'batch_h': triplets[:, 0].to(self.device),
                'batch_t': triplets[:, 2].to(self.device),
                'batch_r': triplets[:, 1].to(self.device),
                'mode': 'normal'
		    }
            tails_predictions = self.model.predict(data).reshape(current_batch_size, -1)
            # Check all possible heads
            triplets = torch.stack((all_entities, relations, tails), dim=2).reshape(-1, 3)
            data = {
                'batch_h': triplets[:, 0].to(self.device),
                'batch_t': triplets[:, 2].to(self.device),
                'batch_r': triplets[:, 1].to(self.device),
                'mode': 'normal'
		    }
            heads_predictions = self.model.predict(data).reshape(current_batch_size, -1)

            # Concat predictions
            predictions = torch.cat((tails_predictions, heads_predictions), dim=0)
            ground_truth_entity_id = torch.cat((tail.reshape(-1, 1), head.reshape(-1, 1)))

            hits_at_1 += hit_at_k(predictions, ground_truth_entity_id, device=self.device, k=1)
            hits_at_3 += hit_at_k(predictions, ground_truth_entity_id, device=self.device, k=3)
            hits_at_10 += hit_at_k(predictions, ground_truth_entity_id, device=self.device, k=10)
            mrr_score += mrr(predictions, ground_truth_entity_id)

            examples_count += predictions.size()[0]

        hits_at_1_score = hits_at_1 / examples_count * 100
        hits_at_3_score = hits_at_3 / examples_count * 100
        hits_at_10_score = hits_at_10 / examples_count * 100
        mrr_score = mrr_score / examples_count * 100
        print(hits_at_1_score, hits_at_3_score, hits_at_10_score, mrr_score)
        return hits_at_1_score, hits_at_3_score, hits_at_10_score, mrr_score