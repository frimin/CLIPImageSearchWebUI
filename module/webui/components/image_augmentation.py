import os
import sys
import gradio as gr
import random
import math
import torch
import module.utils.run_util as run_util
from tqdm import tqdm
from module.foundation.webui import TopElements
from module.foundation.image_augmentation import (
    ImageAugmentationPipeline, 
    divide_chunks, 
    STEP_ITEMS, 
    MAX_STEP,
    ImageAugmentationActions, 
    PresetsList
)
from module.data import get_cache_root, get_webui_configs, get_yolos_model, get_yolo_model_list
from omegaconf import OmegaConf, ListConfig, DictConfig
from module.core.src_datasets import SrcDataset
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

def on_to_yaml_config(crop_pixel_min, crop_scale_min, crop_scale_max, skip_process_pixel, *args):
    global_opt = DictConfig({
        "crop_pixel_min": int(crop_pixel_min),
        "crop_scale_min": crop_scale_min,
        "crop_scale_max": crop_scale_max,
        "skip_pixel": int(skip_process_pixel),
    }) 

    actions = []

    for action_index, get_input, send_output, save_file, force_continue in divide_chunks(args, STEP_ITEMS):
        attributes = []
        if get_input:
            attributes.append("input")
        if send_output:
            attributes.append("output")
        if save_file:
            attributes.append("save")
        if force_continue:
            attributes.append("continue")

        action_info = ImageAugmentationActions._actions[action_index]

        act = DictConfig({
            "id": action_info["id"],
            "attributes": " ".join(attributes)
        })

        actions.append(act)

    cfg = DictConfig({
        "opt": global_opt,
        "actions": ListConfig(actions)
    })

    text = OmegaConf.to_yaml(cfg)
    return text

def on_process_yaml_pipeline(indir, outdir, outdir_with_step, num_workers, yaml_str):
    if not indir or not os.path.exists(indir):
        raise gr.Error(f"输入目录不存在: {indir}")
    if not outdir:
        raise gr.Error(f"无效的输出目录: {outdir}")

    assert yaml_str is not None

    temp_yaml_filename = get_cache_root().create_temporary_filename("yaml")

    with open(temp_yaml_filename, "w") as f:
        f.write(yaml_str)

    args = [ 
        'scripts/create_augmentation.py',
        f'"{temp_yaml_filename}"',
        '--indir', f'"{indir}"',
        '--outdir', f'"{outdir}"',
        '--num_workers', int(num_workers),
    ]

    if outdir_with_step:
        args.append("--outdir_with_step")

    run_cmd = sys.executable + " " + " ".join([str(i) for i in args] )

    print(run_cmd)

    run_util.run(run_cmd) 

    return "完成"

def on_load():
    PresetsList.init()
    return gr.Dropdown.update(choices=[i[0] for i in PresetsList.value])

def on_load_presets(index):
    _, filename = PresetsList.value[index]

    with open(filename, "r") as f:
        return f.read()

def create_image_augmentation(top_elems: TopElements):
    yolo_model_list = get_yolo_model_list()

    with gr.Blocks() as block:
        ImageAugmentationActions.init(yolo_model_list)

        drop_desc = [i["desc"] for i in ImageAugmentationActions._actions if i["visible"]]

        with gr.Row():
            indir = gr.Textbox(label="输入目录", scale=2)
            default_save_path = os.path.join(Path.home(), "Pictures", "CLIPImageSearchWebUI", "Augmentation")
            outdir = gr.Textbox(label="输出目录", value=default_save_path, scale=2)
            outdir_with_step = gr.Checkbox(label="输出至子目录中", info="为每个步骤分别建立子级目录进行输出", value=True, interactive=True)
            num_workers = gr.Number(label="工作进程数量", value=8, minimum=1, maximum=64)

        with gr.Tabs() as tabs:
            with gr.TabItem(label="普通模式", id=0):
                with gr.Row():
                    convert_to_yaml_cfg = gr.Button("转换到配置")
                    generate_btn = gr.Button("生成", variant="primary")

                with gr.Row():
                    skip_process_pixel = gr.Number(label="跳过图片", info="长或宽低于此分辨率的图像忽略处理", value=512, interactive=True)
                    crop_pixel_min = gr.Number(label="裁剪最小边", value=512, info="裁剪结果当最小边小于此值时，将丢弃输出", interactive=True, minimum=64)
                    crop_scale_min = gr.Slider(label="裁剪起始比例", value=0.5, minimum=0.01, maximum=1.0, step=0.01, interactive=True)
                    crop_scale_max = gr.Slider(label="裁剪结束比例", value=0.5, minimum=0.01, maximum=1.0, step=0.01, interactive=True)

                components = []

                for i in range(1, MAX_STEP + 1):
                    with gr.Row():
                        pipe_action = gr.Dropdown(label=f"步骤 {i}", choices=drop_desc, scale=5, type="index", value=0)
                        get_input = gr.Checkbox(label="接收输入", value=True, interactive=True)
                        send_output = gr.Checkbox(label="发送输出", value=True, interactive=True)
                        save_file = gr.Checkbox(label="保存文件", value=True, interactive=True)
                        force_continue = gr.Checkbox(label="强制继续", value=True, interactive=True)

                        components += [pipe_action, get_input, send_output, save_file, force_continue]

            with gr.TabItem(label="配置模式", id=1):
                with gr.Row():
                    presets_dropdown = gr.Dropdown(label="预设", scale=3, interactive=True, type="index")
                    load_presets = gr.Button("加载预设")
                with gr.Row():
                    generate_from_yaml_btn = gr.Button("从配置生成", variant="primary")
                pipline_yaml = gr.Code(label="Pipline YAML", language="yaml", interactive=True)

        internal_pipline_yaml = gr.State()

    convert_to_yaml_cfg.click(on_to_yaml_config, [crop_pixel_min, crop_scale_min, crop_scale_max, skip_process_pixel] + components, [pipline_yaml]).then(lambda : gr.Tabs.update(selected=1), [], [tabs])
    generate_btn.click(on_to_yaml_config, [crop_pixel_min, crop_scale_min, crop_scale_max, skip_process_pixel] + components, [internal_pipline_yaml]).then(
        on_process_yaml_pipeline, 
        [indir, outdir, outdir_with_step, num_workers, internal_pipline_yaml], 
        [top_elems.msg_text]
    )
    generate_from_yaml_btn.click(on_process_yaml_pipeline, [indir, outdir, outdir_with_step, num_workers, pipline_yaml], [top_elems.msg_text])

    block.load(on_load, [], [presets_dropdown])
    load_presets.click(on_load_presets, [presets_dropdown], [pipline_yaml])