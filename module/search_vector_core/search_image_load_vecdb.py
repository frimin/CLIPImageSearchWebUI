import gradio as gr
from module.search_vector_core.search_state import SearchVectorPageState
import module.utils.constants_util as constants_util
from tqdm import tqdm
import os

import gradio as gr
from module.search_vector_core.search_state import SearchVectorPageState
import module.utils.constants_util as constants_util
from module.data import get_vector_db_mgr, get_clip_model, get_cache_root
from tqdm import tqdm
import os
import json
from module.webui.components.search_target import on_load_search_target, get_search_target

_reload_db_index = 0

def on_gui():
    with gr.Tab(label="重新加载"):
        gr.Markdown("加载当前查询结果为新的向量库")
        vecdb_name = gr.Textbox(label="名称", info="为空则分配一个名称" , value="", interactive=True)
        load_btn = gr.Button(value="加载", variant="primary")

    return [vecdb_name, load_btn]

def on_bind(search_state: SearchVectorPageState, compolents: list[gr.components.Component]):
    def on_load_vecdb(vecdb_name: str,
                      search_target,
                      page_state: dict(), 
                      progress = gr.Progress(track_tqdm=True)):
        global _reload_db_index

        vector_mgr = get_vector_db_mgr()
        clip_model = get_clip_model()

        vecdb_name = vecdb_name.strip()

        if not vecdb_name:
            vecdb_name = f"从查询结果加载_{_reload_db_index}" 
            _reload_db_index+=1
        
        if not page_state:
            raise constants_util.INVALID_QUERT_RECORD_ERROR

        search_id = page_state["search_id"]

        cache_root = os.path.join(get_cache_root().cache_root, "search_id", search_id)

        if not os.path.isdir(cache_root):
            raise constants_util.INVALID_QUERT_RECORD_ERROR

        if vector_mgr.is_exsits_variant(vecdb_name):
            raise gr.Error("该库名称已经被使用")

        with open(os.path.join(cache_root, "pages_index.json"), "r") as f:
            page_info = json.load(f)

        save_db_root = os.path.join(cache_root, "vecdb")

        for sub_db_path in os.listdir(save_db_root):
            db_path = os.path.join(save_db_root, sub_db_path)
            vector_mgr.load_path(clip_model=clip_model, db_path=db_path, image_root=None, variant=vecdb_name)

        for name in vector_mgr.get_master_names():
            if name not in search_target["db"]:
                search_target["db"].append(name)
        
        return [
            # vecdb_name
            "", 
            # msg_text
            "", 
        ]

    search_target_inputs_outputs = get_search_target()

    vecdb_name, load_btn = compolents

    load_btn.click(fn=on_load_vecdb, inputs=[
        vecdb_name,
        search_state.search_target,
        search_state.page_state,
    ], outputs=[
        vecdb_name,
        search_state.msg_text,
    ]).then(on_load_search_target, search_target_inputs_outputs, search_target_inputs_outputs)