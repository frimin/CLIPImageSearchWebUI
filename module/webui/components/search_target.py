import gradio as gr
from module.data import get_vector_db_mgr

SEARCH_TARGET_INPUTS_AND_OUTPUTS = None

def set_search_target(list):
    global SEARCH_TARGET_INPUTS_AND_OUTPUTS
    SEARCH_TARGET_INPUTS_AND_OUTPUTS = list

def get_search_target():
    return SEARCH_TARGET_INPUTS_AND_OUTPUTS

def on_load_search_target(search_target, select_search_target):
    vector_mgr = get_vector_db_mgr()

    if search_target is None:
        search_target = { "db": [] }

    for name in vector_mgr.get_master_names():
        if name not in search_target["db"]:
            search_target["db"].append(name)

    if select_search_target is None or len(select_search_target) == 0:
        if "default" in search_target["db"]:
            select_search_target = [ "default" ]
        else:
            if len(search_target["db"]) == 0:
                select_search_target = None
            else:
                select_search_target = [ search_target["db"][0] ]

    return (
        #search_target
        search_target,
        #select_search_target
        gr.Dropdown.update(choices=search_target["db"], value=select_search_target)
        )