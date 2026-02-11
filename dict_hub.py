import os
import glob
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer

from config import args
from triplet import TripletDict, EntityDict, LinkGraph, CoTailGraph , DynamicCache
from logger_config import logger

train_triplet_dict: TripletDict = None
all_triplet_dict: TripletDict = None
link_graph: LinkGraph = None
entity_dict: EntityDict = None
tokenizer: AutoTokenizer = None

cotail_graph :CoTailGraph=None
dynamic_cache :DynamicCache = None

def _init_dynamic_cache():
    global dynamic_cache
    if not dynamic_cache:
        dynamic_cache = DynamicCache()
def get_dynamic_cache():
    _init_dynamic_cache()
    return dynamic_cache

def _init_cotail_graph_valid():
    global cotail_graph
    if not cotail_graph:
        cotail_graph = CoTailGraph(train_path=args.valid_path)

def get_cotail_graph_valid():
    _init_cotail_graph_valid()
    return cotail_graph

def _init_cotail_graph():
    global cotail_graph
    if not cotail_graph:
        cotail_graph = CoTailGraph(train_path=args.train_path)

def get_cotail_graph():
    _init_cotail_graph()
    return cotail_graph
def _init_entity_dict():
    global entity_dict
    if not entity_dict:
        entity_dict = EntityDict(entity_dict_dir=os.path.dirname(args.valid_path))

def _init_train_triplet_dict():
    global train_triplet_dict
    if not train_triplet_dict:
        train_triplet_dict = TripletDict(path_list=[args.train_path])

def _init_all_triplet_dict():
    global all_triplet_dict
    if not all_triplet_dict:
        path_pattern = '{}/*.txt.json'.format(os.path.dirname(args.train_path))
        all_triplet_dict = TripletDict(path_list=glob.glob(path_pattern))

def _init_link_graph():
    global link_graph
    if not link_graph:
        link_graph = LinkGraph(train_path=args.train_path)

def get_entity_dict():
    _init_entity_dict()
    return entity_dict

def get_train_triplet_dict():
    _init_train_triplet_dict()
    return train_triplet_dict

def get_all_triplet_dict():
    _init_all_triplet_dict()
    return all_triplet_dict

def get_link_graph():
    _init_link_graph()
    return link_graph

def build_tokenizer(args):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        logger.info('Build tokenizer from {}'.format(args.pretrained_model))

def get_tokenizer():
    if tokenizer is None:
        build_tokenizer(args)
    return tokenizer