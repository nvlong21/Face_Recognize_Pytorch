from .proxyless_nas import model_zoo
import torch
import torch.nn as nn
def ProxyNas(num_feauture = 512):
	net = model_zoo.proxyless_base()
	net.classifier = nn.Linear(1432, num_feauture, bias = True)
	return net
