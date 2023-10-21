import gradio as gr
from module.foundation.webui import TopElements, TopTabContext
from module.data import get_webui_configs

def create_top(top_tab_context: TopTabContext) -> TopElements:
    top_elems = TopElements()
    top_elems.msg_text = gr.Textbox(value="", label="消息", interactive=False)
    top_elems.top_tab_context = top_tab_context
    return top_elems