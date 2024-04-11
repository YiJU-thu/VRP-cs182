import torch
from torch import nn

from loguru import logger

from nets.encoder_gat import AttentionEncoder
from nets.decoder_gat import AttentionDecoder
from nets.decoder_nAR import NonAutoRegDecoder

import time

class VRPModel(nn.Module):

    encoders = {
        "gat": AttentionEncoder,
        "gcn": None
    }

    decoders = {
        "gat": AttentionDecoder,
        "nAR": NonAutoRegDecoder,
    }

    def __init__(self, encoder_name, decoder_name, encoder_kws, decoder_kws):
        super(VRPModel, self).__init__()

        self.encoder_name = encoder_name
        self.decoder_name = decoder_name

        # avoid en/decoders being called directly outside the model
        self._encoder = self.encoders[encoder_name](**encoder_kws)
        self._decoder = self.decoders[decoder_name](**decoder_kws)
        # NOTE: do not change attr names if the model params are to be loaded from a file
        # so in old versions, as I use self.encoder/decoder, I still need these names to have the params correctly loaded
        self.encoder, self.decoder = self._encoder, self._decoder



        self.time_count = {
            "encoder_forward": 0, "decoder_forward": 0, "model_update": 0, "data_gen": 0, "baseline_eval": 0,
        }


        self.rescale_dist = self._encoder.rescale_dist
        self.non_Euc = self._encoder.non_Euc
        self.problem = self._encoder.problem


    
    def forward(self, input, **kws):
        """
        :param input: (batch_size, graph_size, node_dim) input node features or dictionary with multiple tensors
        :return:
        FIXME: in the notes, make it clear what **kw probably are
        """

        t0 = time.perf_counter()
        embed = self._encoder(input)    # embed is a dict, keys specific to the encoder & compatible with the decoder
        embed.update(kws)

        elif self.decoder_name == "nAR":
            heatmap = self.encoder(input)
            t1 = time.perf_counter()
        res = self._decoder(input, **embed)


        # NOTE: old version, no longer needed
        # if self.decoder_name == "gat":
        #     embeddings, graph_embed = self._encoder(input)
        #     t1 = time.perf_counter()
        #     res = self._decoder(input, embeddings, graph_embed=graph_embed, **kws)
            
        # elif self.decoder_name == "nAR":
        #     heatmap = self._encoder(input)
        #     t1 = time.perf_counter()
        #     res = self._decoder(input, heatmap, **kws)
        
        t2 = time.perf_counter()
        self.update_time_count(encoder_forward=t1-t0, decoder_forward=t2-t1)
        
        return res

    
    def set_decode_type(self, decode_type, temp=None):
        self.decoder.set_decode_type(decode_type, temp)
    
    def update_time_count(self, **kw):
        for key in kw:
            self.time_count[key] += kw[key]
    
    @property
    def time_stats(self):
        total_time = sum(self.time_count.values())
        if total_time == 0:
            return {f'T-{key}': 0 for key in self.time_count}
        return {f'T-{key}': self.time_count[key] / total_time for key in self.time_count}