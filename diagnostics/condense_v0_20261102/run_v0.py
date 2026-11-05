"""Standalone V0 evidence; run from the project root. No real datasets used."""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from torch_geometric.data import Data
from openfgl.flcore.multimodal_fedgvd.SimGC_transductive import data_gc, _adjacency
from openfgl.flcore.multimodal_fedgvd.fedgvd_config import config
from openfgl.flcore.multimodal_fedgvd.utils import cache_signature


def fixture():
    g = torch.Generator().manual_seed(42)
    n, classes = 200, 4
    y = torch.arange(n) % classes
    text = torch.randn(classes, 64, generator=g)[y] + .3 * torch.randn(n, 64, generator=g)
    image = torch.randn(classes, 128, generator=g)[y] + .3 * torch.randn(n, 128, generator=g)
    mask = torch.ones(n, 2, dtype=torch.bool)
    missing = torch.randperm(n, generator=g)[:20]
    mask[missing, torch.arange(20) % 2] = False
    text[~mask[:, 0]], image[~mask[:, 1]] = 0, 0
    # Undirected random graph, predominantly within class, without val/test labels in optimization.
    parts = []
    for i in range(n):
        candidates = torch.where(y == y[i])[0]
        targets = candidates[torch.randperm(len(candidates), generator=g)[:6]]
        parts.extend((i, int(j)) for j in targets if i != int(j))
    edges = torch.tensor(parts).T
    edges = torch.cat((edges, edges.flip(0)), dim=1).unique(dim=1)
    data = Data(x=torch.cat((image, text), 1), x_text=text, x_image=image,
                y=y, edge_index=edges, modality_mask=mask, num_global_classes=classes)
    return dict(data=data, train_mask=torch.arange(n) < 120,
                val_mask=(torch.arange(n) >= 120) & (torch.arange(n) < 160),
                test_mask=torch.arange(n) >= 160)


def main():
    torch.set_num_threads(2)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'V0 torch={torch.__version__} device={device} seed=42 nodes=200 dims=64/128 classes=4 missing=10%', flush=True)
    saved = dict(config)
    rows = []
    try:
        for mode, alternating in [('legacy', False), ('consistent_low', True), ('soft', True)]:
            config.update(adj_mode=mode, alternating_optim=alternating)
            processed = fixture()
            # A new empty cache directory per mode; data_gc must condense, never reuse old .pt.
            with tempfile.TemporaryDirectory(prefix=f'v0_{mode}_', dir=Path(__file__).parent) as tmp:
                args = SimpleNamespace(dataset=['Movies'], seed=42, num_clients=10,
                    hid_dim=32, num_layers=2, dropout=0., lr=.01, weight_decay=0.,
                    multimodal_reduction_rate=.2, multimodal_teacher_epochs=50,
                    multimodal_condense_epochs=100, multimodal_student_epochs=100,
                    multimodal_validation_interval=25, multimodal_fusion_temperature=.5,
                    multimodal_condensed_pattern=str(Path(tmp) / '{dataset}_{seed}_{client_id}.pt'))
                print(f'\n=== {mode} alternating={alternating} teacher=50 condense=100 student=100 ===', flush=True)
                captured = io.StringIO()
                class Tee:
                    def write(self, value):
                        sys.__stdout__.write(value)
                        captured.write(value)
                    def flush(self):
                        sys.__stdout__.flush()
                with contextlib.redirect_stdout(Tee()):
                    graph = data_gc(processed, 0, args, device)
                output = captured.getvalue()
                accuracies = list(map(float, re.findall(r'teacher_on_syn_acc=([\d.]+)', output)))
                counts = list(map(int, re.findall(r'retained_edges=(\d+)', output)))
                shifts = re.search(r'final_text_relative_shift=([\d.]+) final_image_relative_shift=([\d.]+)', output)
                assert graph.num_nodes == 24 and graph.num_edges > 0
                assert torch.isfinite(graph.edge_weight).all() and not graph.x_text.requires_grad
                assert len(accuracies) == 5 and len(counts) == 5 and min(counts) > 0
                assert output.count('weight.mean=') == 4 and output.count('fraction_ge_adj_threshold=') == 4
                assert shifts is not None and min(map(float, shifts.groups())) > 0
                assert accuracies[-1] > accuracies[0], f'{mode}: teacher-on-syn did not rise: {accuracies}'
                rows.append(dict(mode=mode, alternating=alternating, teacher_initial=accuracies[0],
                    teacher_final=accuracies[-1], selected_edges=graph.num_edges,
                    text_shift=float(shifts[1]), image_shift=float(shifts[2])))
                print(f'V0 PASS {mode}', flush=True)
        # Explicit topology semantics and differentiability, including low-score edges.
        edges = torch.tensor([[0, 0, 1], [1, 2, 2]])
        w = torch.tensor([.005, .02, .7], requires_grad=True)
        for mode, expected in [('consistent_low', (2, 2)), ('soft', (3, 3)), ('legacy', (3, 1))]:
            for validation, count in zip((False, True), expected):
                e, weights = _adjacency(edges, w, mode, validation)
                assert e.size(1) == count
                assert weights.requires_grad
        _adjacency(edges, w, 'consistent_low')[1].sum().backward()
        torch.testing.assert_close(w.grad, torch.tensor([0., 1., 1.]))
        p = fixture()
        settings = {'config': dict(config)}
        sig = cache_signature(p, settings)
        assert 'condense_version' not in settings
        assert sig != cache_signature(p, {**settings, 'config': {**config, 'adj_mode': 'legacy'}})
        print('PASS topology train/eval policy, retained-weight gradients, config signature, caller settings unchanged', flush=True)
        Path(__file__).with_name('summary.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(json.dumps(rows, indent=2), flush=True)
        print('V0 ALL PASS', flush=True)
    finally:
        config.clear()
        config.update(saved)


if __name__ == '__main__':
    main()
