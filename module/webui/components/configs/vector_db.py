import gradio as gr
from omegaconf import ListConfig
from module.foundation.webui import TopElements, ConfigUIBuilder, VALUE_TYPE_INT
from module.data import get_webui_configs, get_vector_db_mgr
from ..search_target import on_load_search_target, get_search_target

def on_show_info_vector_db():
    vector_db = get_vector_db_mgr()
    lines = []

    i = 0
    for k, v in vector_db.vecdb_variants.items():
        lines.append(f"db.{i}.name={k}")
        lines.append(f"db.{i}.entity_count={v.get_entity_count()}")
        for fs_i, fs_path in enumerate(v.loaded_path):
            lines.append(f"db.{i}.fs_root.{fs_i}={fs_path}")
        i+=1

    if i == 0:
        lines.append("(no_db_loaded)")

    return '\n'.join(lines)

def on_load_vector_db(progress=gr.Progress(track_tqdm=True)):
    cfg = get_webui_configs().get_cfg()
    vector_db = get_vector_db_mgr()
    vector_db.load_vector_db_from_lines(cfg.librarys)
    return "已加载"

def on_unload_vector_db():
    vector_db = get_vector_db_mgr()
    vector_db.unload_all()

def create_vector_db_configs(top_elems: TopElements, builder: ConfigUIBuilder):
    #vector_db_engine = gr.Dropdown(label="向量数据库引擎", choices=["FAISS (本机)"], value=0, type="index", interactive=False)
    builder.add_elems(gr.Number(label="启动时加载前N个向量库", info="0为加载所有, -1为不加载"), "vector_db.load_start_count", value_type=VALUE_TYPE_INT)

    librarys = gr.TextArea(label="向量库路径, 一行一个，行首添加 <别名> 形式进行具名加载。", interactive=True)

    def get_librarys(cfg):
        cfg = get_webui_configs().get_cfg()
        return '\n'.join(cfg.librarys)

    def set_librarys(cfg, value):
        return ListConfig([i.strip() for i in value.split("\n") if len(i.strip()) > 0])

    builder.add_elems(librarys, "librarys", get_librarys, set_librarys)

    with gr.Accordion(label="辅助功能"):
        info = gr.Code(label="向量库信息", visible=False)

        with gr.Row():
            info_clip_model = gr.Button("查看已加载的向量库信息")
            info_clip_model.click(on_show_info_vector_db, [], [info]).then(lambda : gr.Accordion.update(visible=True), [], [info])

            search_target_inputs_outputs = get_search_target()

            load_clip_model = gr.Button("加载向量数据库")
            load_clip_model.click(
                on_load_vector_db, [], [top_elems.msg_text]
            ).then(on_load_search_target, search_target_inputs_outputs, search_target_inputs_outputs)

            unload_clip_model = gr.Button("卸载正在使用中的向量数据库")
            unload_clip_model.click(
                on_unload_vector_db
            ).then(lambda : "已卸载", outputs=[top_elems.msg_text])
