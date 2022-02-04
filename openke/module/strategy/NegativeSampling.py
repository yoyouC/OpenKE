import imp

from zmq import device
from .Strategy import Strategy
import torch

class NegativeSampling(Strategy):

	def __init__(self, model = None, loss = None, batch_size = 256, regul_rate = 0.0, l3_regul_rate = 0.0, device = None, tot_ent=0):
		super(NegativeSampling, self).__init__()
		self.model = model
		self.loss = loss
		self.batch_size = batch_size
		self.regul_rate = regul_rate
		self.l3_regul_rate = l3_regul_rate
		self.device = device
		self.tot_ent = tot_ent

	def _get_positive_score(self, score, size):
		positive_score = score[:size]
		positive_score = positive_score.view(-1, size).permute(1, 0)
		return positive_score

	def _get_negative_score(self, score, size):
		negative_score = score[size:]
		negative_score = negative_score.view(-1, size).permute(1, 0)
		return negative_score

	def forward(self, data):
		data = self.negsam(data)
		size = len(data) / 2

		data = {
			'batch_h': data[:, 0].to(self.device),
			'batch_t': data[:, 2].to(self.device),
			'batch_r': data[:, 1].to(self.device),
			'mode': 'normal'
		}
		score = self.model(data)
		p_score = self._get_positive_score(score, int(size))
		n_score = self._get_negative_score(score, int(size))
		loss_res = self.loss(p_score, n_score)
		if self.regul_rate != 0:
			loss_res += self.regul_rate * self.model.regularization(data)
		if self.l3_regul_rate != 0:
			loss_res += self.l3_regul_rate * self.model.l3_regularization()
		return loss_res
	
	def negsam(self, data):
		hs, rs, ts = data
		head_or_tail = torch.randint(high=2, size=hs.size())
		random_entities = torch.randint(high=self.tot_ent, size=hs.size())
		broken_heads = torch.where(head_or_tail == 1, random_entities, hs)
		broken_tails = torch.where(head_or_tail == 0, random_entities, ts)

		pos_triplets = torch.stack((hs, rs, ts), dim=1)
		neg_triplets = torch.stack((broken_heads, rs, broken_tails), dim=1)

		return torch.cat((pos_triplets, neg_triplets), 0)
