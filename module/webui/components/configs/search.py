import gradio as gr
from module.foundation.webui import TopElements, ConfigUIBuilder, VALUE_TYPE_INT
from module.data import get_clip_model_list

def create_search_configs(top_elems, builder: ConfigUIBuilder):
    kwargs = { "interactive": True }

    with gr.Row():
        builder.add_elems(gr.Number(label="默认 TopK", **kwargs), "search.default_top_k", value_type=VALUE_TYPE_INT)
        builder.add_elems(gr.Number(label="TopK 上限", **kwargs), "search.max_top_k", value_type=VALUE_TYPE_INT)
        builder.add_elems(gr.Number(label="默认分页大小", **kwargs), "search.default_page_size", value_type=VALUE_TYPE_INT)
        builder.add_elems(gr.Number(label="最大分页大小", **kwargs), "search.max_page_size", value_type=VALUE_TYPE_INT)

    with gr.Row():
        builder.add_elems(gr.Number(label="图像最大的边超过此尺寸时创建缓存", **kwargs), "cache.image.max_size", value_type=VALUE_TYPE_INT)
        builder.add_elems(gr.Number(label="图片文件超过此大小时创建缓存", **kwargs), "cache.image.greater_than_size", value_type=VALUE_TYPE_INT)