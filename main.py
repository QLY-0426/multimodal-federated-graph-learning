import openfgl.config as config

from openfgl.flcore.trainer import FGLTrainer
from openfgl.utils.basic_utils import seed_everything

# Default run: multimodal_fedtad on Movies; CLI arguments override these defaults.
args = config.args

if args.root == "change_to_your_root_path":
    args.root = "dataset"


is_multimodal = args.fl_algorithm in ("multimodal_fedgvd", "multimodal_fedtad", "multimodal_fedpub")
args.dataset = args.dataset or (["Movies"] if is_multimodal else ["Computers"])
if args.fl_algorithm == "fedgvd":
    args.simulation_mode = "subgraph_fl_louvain"
    args.num_clients = 10


# if True:
#     args.fl_algorithm = "fedavg"
#     args.model = ["gcn"]
# else:
default_models = ["gcn", "gat", "sgc", "mlp", "graphsage"]
if args.fl_algorithm == "multimodal_fedgvd":
    # default_models = ["multimodal_" + name for name in default_models]
    default_models = ["multimodal_gcn"]
elif args.fl_algorithm in ("multimodal_fedtad", "multimodal_fedpub"):
    default_models = ["multimodal_gcn"]
elif args.fl_algorithm != "fedgvd":
    default_models = ["gcn"]
args.model = args.model or default_models
# args.model = ["gcn","gat", "sgc","graphsage"] # choose multiple gnn models for model heterogeneity setting.
# args.model = ["gcn","gat"] # choose multiple gnn models for model heterogeneity setting.
# args.model = ["gcn"] # choose multiple gnn models for model heterogeneity setting.

args.metrics = args.metrics or ["accuracy"]

if is_multimodal:
    if config.unknown:
        config.parser.error("unrecognized arguments: " + " ".join(config.unknown))
    if args.task != "node_cls" or args.scenario != "subgraph_fl" or len(args.dataset) != 1:
        config.parser.error("multimodal algorithms require one subgraph_fl node_cls dataset per run")
    if args.processing != "raw" or args.dp_mech != "no_dp":
        config.parser.error("multimodal algorithms currently support raw processing and no_dp")
    if not 0 < args.client_frac <= 1 or int(args.num_clients * args.client_frac) < 1:
        config.parser.error("client_frac must select at least one client and be <= 1")
    if min(args.num_epochs, args.multimodal_server_epochs, args.num_rounds) < 1:
        config.parser.error("training epoch and round counts must be positive")
    if args.fl_algorithm == "multimodal_fedtad" and len(args.model) != 1:
        config.parser.error("multimodal_fedtad requires a single shared --model, default multimodal_gcn")
    if args.fl_algorithm == "multimodal_fedpub":
        if len(args.model) != 1 or args.model[0] not in ("gcn", "multimodal_gcn") or args.num_layers != 2:
            config.parser.error("multimodal_fedpub requires one shared MaskedGCN: --model multimodal_gcn --num_layers 2")
        if args.evaluation_mode not in ("local_model_on_local_data", "local_model_on_global_data"):
            config.parser.error("multimodal_fedpub is personalized and requires local_model evaluation")

if is_multimodal or args.fl_algorithm == "fedtad":
    seed_everything(args.seed)

trainer = FGLTrainer(args)

trainer.train()
