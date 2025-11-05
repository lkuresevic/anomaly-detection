from torch import nn
from models.encoder import Encoder
from models.attribute_decoder import Attribute_Decoder
from models.structure_decoder import Structure_Decoder

class Dominant(nn.Module):
    def __init__(self, feat_size, hidden_size, dropout, layer_type='gcn'):
        super(Dominant, self).__init__()
        self.layer_type = layer_type.lower()
        self.shared_encoder = Encoder(feat_size, hidden_size, dropout, layer_type=self.layer_type)
        self.attr_decoder = Attribute_Decoder(feat_size, hidden_size, dropout, layer_type=self.layer_type)
        self.struct_decoder = Structure_Decoder(hidden_size, dropout, layer_type=self.layer_type)

    def forward(self, x, adj):
        z = self.shared_encoder(x, adj)
        x_hat = self.attr_decoder(z, adj)
        struct_reconstructed = self.struct_decoder(z, adj)
        return struct_reconstructed, x_hat