import gradio as gr
from module.search_vector_core.search_state import SearchVectorPageState
import module.utils.constants_util as constants_util
from tqdm import tqdm
from module.data import get_webui_configs
from module.utils.constants_util import DISABLE_DELETE_FILE_ERROR

import module.utils.path_util as path_util
import os
import json

model = None

def on_gui():
    with gr.Tab(label="查询删除"):
        with gr.Row():
            input_search_id = gr.Textbox(label="删除此查询ID原始文件", info="为了防止误操作，必须先获取当前查询ID，再删除指定查询ID相关的所有原始文件" , value="", interactive=True)
        with gr.Row():
            delete_same_name_ext = gr.Textbox(label="删除同名文件", info="删除同名的其它后缀文件" , value=".txt,.caption,.json", interactive=True)

        show_search_id_btn = gr.Button(value="获取当前查询ID")
        delete_btn = gr.Button(value="删除指定查询ID的所有原始文件 (慎重,文件不可恢复)", variant="primary")

    return [input_search_id, delete_same_name_ext, show_search_id_btn, delete_btn]

def on_bind(search_state: SearchVectorPageState, compolents: list[gr.components.Component]):
    def on_show_search_id(page_state: dict(), select_search_history: str):
        if not page_state:
            raise constants_util.INVALID_QUERT_RECORD_ERROR
        return page_state["search_id"] + "@" + select_search_history

    def on_delete(search_id: str,
                  delete_same_name_ext: str,
                  progress = gr.Progress(track_tqdm=True)):

        if get_webui_configs().get_cfg().security.disalbe_delete_file:
            raise DISABLE_DELETE_FILE_ERROR 

        pos_idx = search_id.find("@")

        if pos_idx >= 0:
            search_id = search_id[:pos_idx]

        del_ext = [i.strip() for i in delete_same_name_ext.split(',') ]
        del_ext += constants_util.IMAGE_EXTENSIONS

        cache_root = os.path.join(path_util.get_cache_dir(), "search_id", search_id)

        if not os.path.isdir(cache_root):
            raise constants_util.INVALID_QUERT_RECORD_ERROR

        with open(os.path.join(cache_root, "pages_index.json"), "r") as f:
            page_info = json.load(f)

        count = 0

        with open(os.path.join(cache_root, "pages.json"), "r") as f:
            for page_pos_start, page_pos_end in tqdm(page_info, desc="删除分页"):
                cur_fd_pos = f.tell()
                if cur_fd_pos != page_pos_start:
                    f.seek(page_pos_start)
                content = f.read(page_pos_end - page_pos_start)
                files = json.loads(content)

                for filename, _ in files:
                    for ext in del_ext:
                        base_name = os.path.basename(filename)
                        filename_with_ext = filename + ext
                        if os.path.exists(filename_with_ext):
                            count += 1
                            print(f"delete: {filename_with_ext}")
                            os.remove(filename_with_ext)
        
        return ["", f"已删除{count}个文件"]

    input_search_id, delete_same_name_ext, show_search_id_btn, delete_btn = compolents

    show_search_id_btn.click(fn=on_show_search_id, inputs=[search_state.page_state, search_state.select_search_history], outputs=[input_search_id])
    
    delete_btn.click(fn=on_delete, inputs=[
            input_search_id,
            delete_same_name_ext,
        ], outputs=[
            input_search_id,
            search_state.msg_text,
        ])