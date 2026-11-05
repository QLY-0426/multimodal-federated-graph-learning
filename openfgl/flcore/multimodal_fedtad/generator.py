from openfgl.flcore.fedtad.generator import FedTAD_ConGenerator


class MultimodalFedTADGenerator(FedTAD_ConGenerator):
    """Convenience wrapper for paired [image, text] output; server reuses the original class."""

    def __init__(self, noise_dim, modality_dims, out_dim, dropout):
        super().__init__(noise_dim, sum(modality_dims), out_dim, dropout)
