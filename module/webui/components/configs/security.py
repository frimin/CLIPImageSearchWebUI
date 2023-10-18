import gradio as gr
from module.foundation.webui import TopElements, ConfigUIBuilder
from module.data import get_clip_model_list

def create_security_configs(top_elems, builder: ConfigUIBuilder):
    kwargs = { "interactive": True }

    with gr.Row():
        builder.add_elems(gr.Checkbox(label="禁用所有文件删除功能", **kwargs), "security.disalbe_delete_file")