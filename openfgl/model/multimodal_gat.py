from .multimodal_utils import DualBranchModel


class MultimodalGAT(DualBranchModel):
    def __init__(self, input_dim, hid_dim, output_dim, num_layers=2, dropout=0.5, fusion_temperature=0.5):
        super().__init__('gat', input_dim, hid_dim, output_dim, num_layers, dropout, fusion_temperature)
