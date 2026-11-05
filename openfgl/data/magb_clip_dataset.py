"""PyG wrapper for MAGB graphs with pre-extracted CLIP features."""

from os import path as osp

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset


class MAGBClipDataset(InMemoryDataset):
    @property
    def raw_file_names(self):
        return [
            "node_ids.npy",
            "image_features.npy",
            "text_features.npy",
            "edge_index.npy",
            "labels.npy",
        ]

    @property
    def processed_file_names(self):
        return "data.pt"

    def __init__(self, root, force_reload=False):
        super().__init__(root, force_reload=force_reload)
        self.load(self.processed_paths[0])

    def process(self):
        def read(name):
            return np.load(osp.join(self.raw_dir, name), allow_pickle=False)

        node_ids = read("node_ids.npy")
        image = read("image_features.npy")
        text = read("text_features.npy")
        edges = read("edge_index.npy")
        labels = read("labels.npy")

        n = len(node_ids)
        if node_ids.ndim != 1 or not np.array_equal(node_ids, np.arange(n)):
            raise ValueError("node_ids must be contiguous and ordered from zero")
        if (image.shape != text.shape or image.ndim != 2
                or image.shape[0] != n or labels.shape != (n,)):
            raise ValueError("image, text, labels, and node_ids are misaligned")
        if edges.ndim != 2 or edges.shape[0] != 2 or edges.size == 0:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if not np.isfinite(image).all() or not np.isfinite(text).all():
            raise ValueError("CLIP features contain non-finite values")
        if labels.min() < 0 or edges.min() < 0 or edges.max() >= n:
            raise ValueError("labels or edges contain invalid indices")

        x_image = torch.from_numpy(image.astype(np.float32, copy=False)).contiguous()
        x_text = torch.from_numpy(text.astype(np.float32, copy=False)).contiguous()
        x = torch.cat([F.normalize(x_image, dim=1), F.normalize(x_text, dim=1)], dim=1)
        data = Data(
            x=x,
            x_image=x_image,
            x_text=x_text,
            edge_index=torch.from_numpy(edges.astype(np.int64, copy=False)).contiguous(),
            y=torch.from_numpy(labels.astype(np.int64, copy=False)).contiguous(),
            node_id=torch.from_numpy(node_ids.astype(np.int64, copy=False)).contiguous(),
            num_nodes=n,
        )
        data.validate(raise_on_error=True)
        self.save([data], self.processed_paths[0])
