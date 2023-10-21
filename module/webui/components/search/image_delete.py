import gradio as gr
from module.search_vector_core.search_state import SearchVectorPageState
import module.utils.constants_util as constants_util
from tqdm import tqdm
from module.data import get_cache_root, get_webui_configs
from module.foundation.webui import TopElements

import module.utils.path_util as path_util
import os
import json

def make_callback(top_elems: TopElements, button, search_state):
    def on_create_delete_list(page_state: dict(), select_search_history: str, progress = gr.Progress(track_tqdm=True)):
        if not page_state:
            raise constants_util.INVALID_QUERT_RECORD_ERROR

        print(select_search_history)

        temp_file = get_cache_root().create_temporary_filename(".txt")

        cache_root = os.path.join(path_util.get_cache_dir(), "search_id", page_state["search_id"])

        if not os.path.isdir(cache_root):
            raise constants_util.INVALID_QUERT_RECORD_ERROR

        total_count = 0

        with open(os.path.join(cache_root, "pages_index.json"), "r") as f:
            page_info = json.load(f)

        with open(temp_file, "w") as temp_f:
            with open(os.path.join(cache_root, "pages.json"), "r") as f:
                for page_pos_start, page_pos_end in tqdm(page_info, desc="记录分页"):
                    cur_fd_pos = f.tell()
                    if cur_fd_pos != page_pos_start:
                        f.seek(page_pos_start)
                    content = f.read(page_pos_end - page_pos_start)
                    files = json.loads(content)

                    for filename, _ in files:
                        image_file = None
                        for ext in constants_util.IMAGE_EXTENSIONS:
                            image_file = filename + ext
                            if os.path.exists(image_file):
                                temp_f.write(image_file + "\n")
                                total_count += 1
                                continue

        return f"已收集到 {total_count} 个图片文件", temp_file

    top_elems.goto_tab(button.click(on_create_delete_list, [
        search_state.page_state, 
        search_state.select_search_history,
    ], [
        top_elems.msg_text,
        top_elems.get_shared_component("file_checker.list_file")
    ]).then, "extensions.file_checker")

def create_image_delete(top_elems: TopElements, search_state: SearchVectorPageState):
    with gr.Tab(label="查询删除"):
        send_to_button = gr.Button("将删除列表发送到 [扩展功能] -> [文件检查器] 中", variant="primary")
        top_elems.delay_bind(make_callback, send_to_button, search_state)

def on_gui():
    with gr.Tab(label="查询删除"):
        with gr.Row():
            input_search_id = gr.Textbox(label="删除此查询ID原始文件", info="为了防止误操作，必须先获取当前查询ID，再删除指定查询ID相关的所有原始文件" , value="", interactive=True)
        with gr.Row():
            delete_same_name_ext = gr.Textbox(label="删除同名文件", info="删除同名的其它后缀文件" , value=".txt,.caption,.json", interactive=True)

        show_search_id_btn = gr.Button(value="获取当前查询ID")
        delete_btn = gr.Button(value="删除指定查询ID的所有原始文件 (慎重,文件不可恢复)", variant="primary")

    return [input_search_id, delete_same_name_ext, show_search_id_btn, delete_btn]
