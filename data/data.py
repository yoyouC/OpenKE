from torch.utils import data
from collections import Counter

def create_mappings(data_path):
    """Creates separate mappings to indices for entities and relations."""
    # counters to have entities/relations sorted from most frequent
    entity_counter = Counter()
    relation_counter = Counter()
    with open(data_path, "r") as f:
        for line in f:
            head, relation, tail = line.strip().split("\t")
            entity_counter.update([head, tail])
            relation_counter.update([relation])
    entity2id = {}
    relation2id = {}
    for idx, (entity, _) in enumerate(entity_counter.most_common()):
        entity2id[entity] = idx
    for idx, (relation, _) in enumerate(relation_counter.most_common()):
        relation2id[relation] = idx
    return entity2id, relation2id


class LRDataset(data.Dataset):

    def __init__(self, data_path):
        self.entity2id, self.relation2id = create_mappings(data_path)
        self.id2entity = {v:k for k, v in self.entity2id.items()}
        self.id2relation = {v:k for k, v in self.entity2id.items()}
        
        with open(data_path, "r") as f:
            self.data = [line.strip().split("\t") for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Returns (head id, relation id, tail id)."""
        h, r, t = self.data[index]
        h_id = self._to_idx(h, self.entity2id)
        r_id = self._to_idx(r, self.relation2id)
        t_id = self._to_idx(t, self.entity2id)
        return h_id, r_id, t_id

    @staticmethod
    def _to_idx(key, mapping):
        return mapping[key]
    
    def get_tot_ent(self):
        return len(self.entity2id)

    def get_tot_rel(self):
        return len(self.relation2id)