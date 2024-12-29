import torch
import torch.nn as nn
from model.dlutils import (PositionalEncoding,In_channel_anomaly,
                         Between_channel_anomaly,TransformerEncoderLayer,TransformerDecoderLayer,TranAD_PositionalEncoding)
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from src.constants import lr
torch.manual_seed(1)
import math
import torch

class Bet_In_Chans(nn.Module):
	def __init__(self, feats):
		super(Bet_In_Chans, self).__init__()
		self.name = 'Bet_In_Chans'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats

		self.pos_encoder = PositionalEncoding(feats, 0.1, self.batch)
		self.encoder_layers1 = In_channel_anomaly(batch=self.batch, scale_factor=64, dim_feedforward=512, dropout=0.1)
		self.encoder_layers2 = Between_channel_anomaly(d_model=self.n_feats, scale_factor=64, dim_feedforward=512, dropout=0.1)
		
		self.norm = nn.LayerNorm(normalized_shape=feats)
		
		self.fuse_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
		self.fuse_weight_1.data.fill_(0.5)

	def forward(self, src):
		src1 = self.pos_encoder(src)

		memory1,In_attn = self.encoder_layers1(src1, src1, src1)
		memory2,Bet_attn = self.encoder_layers2(src, src ,src)

		memory = self.fuse_weight_1*memory1+(1-self.fuse_weight_1)*memory2

		return memory,memory1,memory2,In_attn,Bet_attn

class TranAD(nn.Module):
	def __init__(self, feats):
		super(TranAD, self).__init__()
		self.name = 'TranAD'
		self.lr = lr
		self.batch = 128
		self.n_feats = feats
		self.n_window = 10
		self.n = self.n_feats * self.n_window
		self.pos_encoder = TranAD_PositionalEncoding(2 * feats, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * feats, nhead=feats, dim_feedforward=16, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * feats, feats), nn.Sigmoid())#全连接层

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src, tgt):
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		return x1, x2
