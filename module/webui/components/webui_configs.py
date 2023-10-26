import gradio as gr
from typing import Callable
from omegaconf import ListConfig
from module.foundation.webui import TopElements, TopTab, ConfigUIBuilder, VALUE_TYPE_INT
from module.data import (
    get_webui_configs, 
)

from .configs import (
    create_model_configs,
    create_vector_db_configs,
    create_search_configs,
    create_security_configs,
)

def get_cfg():
    return get_webui_configs().get_changes_cfg()

def get_cfg_value_with_path(cfg, path_str):
    paths = path_str.split(".")
    for i in paths:
        if i in cfg:
            cfg = cfg[i]
            continue
        if hasattr(cfg, i):
            cfg = getattr(cfg, i)
            continue
        raise ValueError("index cfg error: " + path_str)
    return cfg

def set_cfg_value_with_path(cfg, path_str, value):
    paths = path_str.split(".")
    latest_path = paths[-1]
    paths = paths[:-1]
    for i in paths:
        if i in cfg:
            cfg = cfg[i]
            continue
        if hasattr(cfg, i):
            cfg = getattr(cfg, i)
            continue
        raise ValueError("index cfg error: " + path_str)
    
    if latest_path in cfg:
        cfg[latest_path] = value 
    elif hasattr(cfg, latest_path):
        setattr(cfg, latest_path, value)

    return cfg

def init_page(page: gr.Blocks, cancel_btn: gr.Button, builder: ConfigUIBuilder):
    outputs = [i["comp"] for i in builder.get_elems()]

    def on_page_load():
        cfg = get_webui_configs().get_cfg()

        init_values = []

        for v in builder.get_elems():
            value_path, get_cb = v["path"], v["get"] 
            if isinstance(get_cb, Callable):
                init_values.append(get_cb(cfg))
            else:
                val = get_cfg_value_with_path(cfg, value_path)
                init_values.append(val)

        return init_values

    page.load(on_page_load, [], outputs)

    cancel_btn.click(on_page_load, [], outputs)

def create_on_save(builder: ConfigUIBuilder):
    inputs = [i["comp"] for i in builder.get_elems()]

    def on_save_all(*inputs):
        cfg = get_cfg()

        is_saved = False

        try:
            for input, v in zip(inputs, builder.get_elems()):
                value_path, set_cb, value_type = v["path"], v["set"], v["type"]
                new_val = None
                if isinstance(set_cb, Callable):
                    new_val = set_cb(cfg, input)
                else:
                    new_val = input

                if value_type == VALUE_TYPE_INT:
                    new_val = int(new_val)

                set_cfg_value_with_path(cfg, value_path, new_val)
                
            is_saved = True
        finally:
            if is_saved:
                get_webui_configs().apply_cfg_changes()
            else:
                get_webui_configs().cancel_cfg_changes()

    return inputs, on_save_all

def create_webui_configs(top_elems: TopElements):
    context = top_elems.top_tab_context

    builder = ConfigUIBuilder()

    """配置页面"""
    with gr.Blocks() as page:
        with gr.Row():
            cancel_btn = gr.Button(value="恢复修改")
            save_btn = gr.Button(value="保存配置", variant="primary")

        gr.Markdown("[GITHUB 在线文档](https://github.com/frimin/CLIPImageSearchWebUI#%E4%BD%BF%E7%94%A8%E8%AF%B4%E6%98%8E)")

        with TopTab.Tabs(context):
            with TopTab.TabItem(context, "model", "模型"):
                create_model_configs(top_elems, builder) 

            with TopTab.TabItem(context, "vectordb", "向量库"):
                create_vector_db_configs(top_elems, builder)

            with TopTab.TabItem(context, "search", "搜索"):
                create_search_configs(top_elems, builder)

            with TopTab.TabItem(context, "security", "安全"):
                create_security_configs(top_elems, builder)

    init_page(page, cancel_btn, builder)

    save_inputs, on_save = create_on_save(builder)

    save_btn.click(on_save, save_inputs).then(lambda: "保存成功", outputs=[top_elems.msg_text])