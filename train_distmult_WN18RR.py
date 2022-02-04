from openke.module.model import DistMult
from openke.module.loss import SoftplusLoss
from openke.module.strategy import NegativeSampling
from utils import Trainer, Tester
from data import LRDataset
import torch
from torch.utils.data import DataLoader

use_gpu = False
device = torch.device('cpu')

test_set = LRDataset('datasets/WN18RR/text/test.txt')
valid_set = LRDataset('datasets/WN18RR/text/valid.txt')
train_set = LRDataset('datasets/WN18RR/text/train.txt')

# define dataloader
test_loader = DataLoader(test_set, batch_size=256, shuffle=False)
valid_loader = DataLoader(valid_set, batch_size=256, shuffle=False)
train_loader = DataLoader(train_set, batch_size=256, shuffle=False)

# define the model
distmult = DistMult(
	ent_tot = train_set.get_tot_ent(),
	rel_tot = train_set.get_tot_rel(),
	dim = 200
)

model = NegativeSampling(
	model = distmult, 
	loss = SoftplusLoss(),
	batch_size = 256, 
	regul_rate = 1.0,
	device=device,
	tot_ent=train_set.get_tot_ent()
)


validator = Tester(model = distmult, data_loader = valid_loader, use_gpu = use_gpu, entity_count = train_set.get_tot_ent())

# train the model
trainer = Trainer(model = model, 
                 data_loader = train_loader, 
                 train_times = 1000, 
                 alpha = 0.5, 
                 use_gpu = use_gpu, 
                 opt_method = "adam",
                 validator=validator,
                 validate_step=10)
trainer.run()
distmult.save_checkpoint('./checkpoint/distmult.ckpt')

# test the model
tester = Tester(model = distmult, data_loader = test_loader, use_gpu = use_gpu, entity_count = train_set.get_tot_ent())
tester.run()