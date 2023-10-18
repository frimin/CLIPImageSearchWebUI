from .cli_args import init_clip_args, get_cli_args
from .webui_configs import init_webui_configs, get_webui_configs, WebUIConfigs
from .clip_model_list import init_clip_model_list, get_clip_model_list, CLIPModelList
from .model.clip import init_clip_model, get_clip_model, load_clip_model, CLIPWarpper
from .db.vectordb import init_vector_db, get_vector_db_mgr, VectorDatabaseManager, VectorDatabase
from .cache import init_cache_root, get_cache_root

def init_data(cli_args):
    init_clip_args(cli_args)
    init_webui_configs(cli_args.userdata_dir) 
    init_cache_root(cli_args.userdata_dir)
    init_clip_model_list()

    webui_configs = get_webui_configs()
    clip_model_list = get_clip_model_list()
    init_clip_model(webui_configs, clip_model_list)
    init_vector_db(webui_configs)
