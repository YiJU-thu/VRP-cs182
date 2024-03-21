import torch
from torch import nn

from nets.encoder_gat import AttentionEncoder
from nets.decoder_gat import AttentionDecoder


class VRPModel(nn.Module):

    encoders = {
        "gat": AttentionEncoder,
        "gcn": None
    }

    decoders = {
        "gat": AttentionDecoder,
        "nAR": None
    }

    def __init__(self, encoder_name, decoder_name, encoder_kws, decoder_kws):
        super(VRPModel, self).__init__()

        self.encoder_name = encoder_name
        self.decoder_name = decoder_name
        self.encoder = self.encoders[encoder_name](**encoder_kws)
        self.decoder = self.decoders[decoder_name](**decoder_kws)
    
    def forward(self, input, **kws):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :return:
        """
        
        if self.decoder_name == "gat":
            embeddings, graph_embed = self.encoder(input)
            return self.decoder(input, embeddings, graph_embed=graph_embed, **kws)
        
        elif self.decoder_name == "nAR":
            heatmap = self.encoder(input)
            return self.decoder(input, heatmap, **kws)
        
        # sampling strategies should be implemented here?
    
    def set_decode_type(self, decode_type, temp=None):
        self.decoder.set_decode_type(decode_type, temp)