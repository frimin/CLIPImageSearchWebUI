import gradio as gr
from module.foundation.webui import TopElements, ConfigUIBuilder
from module.data import get_clip_model_list, get_clip_model

def on_show_info_clip_model():
    clip_model = get_clip_model()
    lines = []

    lines.append(f"clip_model_id={clip_model.clip_model_id}")
    lines.append(f"clip_model_project_url={clip_model.get_project_url()}")
    lines.append(f"short_name={clip_model.get_short_name()}")
    lines.append(f"subpath_name={clip_model.get_model_subpath()}")
    lines.append(f"text_embed_dim={clip_model.get_dim()}")
    
    if clip_model.is_load_model():
        with clip_model.get_model() as m:
            pytorch_total_params = sum(p.numel() for p in m.model.parameters())
            lines.append(f"pytorch_total_params={pytorch_total_params}")
            lines.append(f"device={m.model.device}")
            lines.append(f"dtype={m.model.dtype}")
    else:
        lines.append("pytorch_total_params=(not_loaded)")
        lines.append("device=(not_loaded)")
        lines.append("dtype=(not_loaded)")

    return '\n'.join(lines)

def on_unload_clip_model():
    clip_model = get_clip_model()
    clip_model.release_model()

def create_clip_configs(top_elems: TopElements, builder: ConfigUIBuilder):
    clip_model_id = gr.Dropdown(label="可选的 CLIP 模型", multiselect=False, interactive=True)

    def get_clip_model_id(cfg):
        model_list = get_clip_model_list()
        all_model_id = [i.model_id for i in model_list.get_cfg()]
        return gr.Dropdown.update(choices=all_model_id, value=cfg.clip_model_id)

    builder.add_elems(clip_model_id, "clip_model_id", get_clip_model_id)

    with gr.Row():
        builder.add_elems(gr.Checkbox(label="强制离线直接从本地缓存加载CLIP模型", info="已经从 huggingface 下载过对应模型则不用每次在线检查", interactive=True), "clip.offline_load")

    with gr.Accordion(label="辅助功能"):
        with gr.Row():
            info_clip_model = gr.Button("查看已加载的模型信息")
            unload_clip_model = gr.Button("卸载正在使用中的模型")
        
        info = gr.Code(label="模型信息", visible=False)

        info_clip_model.click(on_show_info_clip_model, [], [info]).then(lambda : gr.Accordion.update(visible=True), [], [info])
        unload_clip_model.click(on_unload_clip_model).then(lambda : "已卸载", outputs=[top_elems.msg_text])

