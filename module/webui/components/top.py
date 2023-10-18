import gradio as gr
from module.foundation.webui import TopElements
from module.data import get_webui_configs

def create_top() -> TopElements:
    top_elems = TopElements()
    top_elems.msg_text = gr.Textbox(value="", label="消息", interactive=False)
    return top_elems