from torch_geometric.nn.pool import *




def load_graph_cls_default_model(args, input_dim, output_dim, client_id=None):
    """
    Load the default model for graph classification tasks.

    Args:
        args (Namespace): Arguments containing model configurations.
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        client_id (int, optional): ID of the client in federated learning. Defaults to None.

    Returns:
        torch.nn.Module: The initialized model.
    """
    if client_id is None: # server
        model_name = args.model[0]
    else: # client
        if len(args.model) > 1:
            model_id = int(len(args.model) * client_id / args.num_clients)
            model_name = args.model[model_id]
        else:
            model_name = args.model[0]
        
            
    if model_name == "gin":
        from openfgl.model.gin import GIN
        return GIN(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "global_edge":
        from openfgl.model.global_edge import GlobalEdge
        return GlobalEdge(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "global_pan":
        from openfgl.model.global_pan import GlobalPAN
        return GlobalPAN(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "global_sag":
        from openfgl.model.global_sag import GlobalSAG
        return GlobalSAG(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)



def load_node_edge_level_default_model(args, input_dim, output_dim, client_id=None):
    """
    Load the default model for node and edge level tasks.

    Args:
        args (Namespace): Arguments containing model configurations.
        input_dim (int): Dimension of the input features.
        output_dim (int): Dimension of the output features.
        client_id (int, optional): ID of the client in federated learning. Defaults to None.

    Returns:
        torch.nn.Module: The initialized model.
    """
    if getattr(args, "fl_algorithm", None) == "multimodal_fedpub":
        from openfgl.flcore.multimodal_fedpub.maskedgcn import MultimodalMaskedGCN
        from openfgl.flcore.multimodal_fedpub.fedpub_config import config
        if len(args.model) != 1 or args.model[0] not in ('gcn', 'multimodal_gcn'):
            raise ValueError('multimodal_fedpub requires one shared MaskedGCN model (--model multimodal_gcn)')
        if args.num_layers != 2:
            raise ValueError('multimodal_fedpub preserves FED-PUB\'s two masked graph convolution layers')
        return MultimodalMaskedGCN(input_dim, args.hid_dim, output_dim, dropout=args.dropout,
            l1=config['l1'], laye_mask_one=config['laye_mask_one'], clsf_mask_one=config['clsf_mask_one'])
    if getattr(args, "fl_algorithm", None) in ("multimodal_fedgvd", "multimodal_fedtad"):
        from importlib import import_module
        if args.fl_algorithm == "multimodal_fedtad":
            if len(args.model) != 1:
                raise ValueError('multimodal_fedtad requires one shared model architecture')
            model_name = args.model[0]
        elif client_id is None:
            model_name = args.multimodal_server_model
        else:
            model_name = args.model[int(len(args.model) * client_id / args.num_clients)]
        model_name = model_name.removeprefix("multimodal_")
        class_names = {"gcn": "MultimodalGCN", "gat": "MultimodalGAT",
                       "sgc": "MultimodalSGC", "mlp": "MultimodalMLP",
                       "graphsage": "MultimodalGraphSAGE"}
        if model_name not in class_names:
            raise ValueError(f"Unsupported multimodal model: {model_name}")
        module = import_module(f"openfgl.model.multimodal_{model_name}")
        model = getattr(module, class_names[model_name])
        result = model(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim,
                     num_layers=args.num_layers, dropout=args.dropout,
                     fusion_temperature=getattr(args, 'multimodal_fusion_temperature', 0.5))
        if args.fl_algorithm == 'multimodal_fedtad':
            result.fusion_mode = 'mean'
            result.input_mode = 'concatenated'
        return result
    if client_id is None: # server
        model_name = args.model[0]
    else: # client
        if len(args.model) > 1:
            model_id = int(len(args.model) * client_id / args.num_clients)
            model_name = args.model[model_id]
        else:
            model_name = args.model[0]
    if model_name == "mlp":
        from openfgl.model.mlp import MLP
        return MLP(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "gcn":
        from openfgl.model.gcn import GCN
        return GCN(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "gat":
        from openfgl.model.gat import GAT
        return GAT(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "graphsage":
        from openfgl.model.graphsage import GraphSAGE
        return GraphSAGE(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "sgc":
        from openfgl.model.sgc import SGC
        return SGC(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    elif model_name == "gcn2":
        from openfgl.model.gcn2 import GCN2
        return GCN2(input_dim=input_dim, hid_dim=args.hid_dim, output_dim=output_dim, num_layers=args.num_layers, dropout=args.dropout)
    else:
        raise ValueError

