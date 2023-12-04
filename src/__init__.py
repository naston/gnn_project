#from .connect import connect_cora,connect_citeseer,connect_imdb,close
from .plotting import plot_deg_dist
from .models import GraphSAGE, DotPredictor, MLPPredictor, GraphEVE, GraphIEVE
from .eval import compute_auc, compute_loss
from .preprocess import create_train_test_split_edge, remove_edges, prepare_node_class
from .train_model import train_link_pred
from .results import run_edge_prediction, run_node_classification, run_joint_train
from .parse_cora_text import get_raw_text_cora#, get_gpt_text_cora
from .parse_pubmed_text import get_raw_text_pubmed
from .parse_arxiv_text import get_raw_text_arxiv

from .lm_trainer import LMTrainer
from .lm_config import cfg, update_cfg