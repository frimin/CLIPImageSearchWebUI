import os
import gradio as gr
import random
import math
import torch
from tqdm import tqdm
from module.foundation.webui import TopElements
from module.data import get_cache_root, get_webui_configs, get_yolos_model, get_yolo_model_list
from omegaconf import OmegaConf, ListConfig, DictConfig
from module.core.src_datasets import SrcDataset
from PIL import Image
from uuid import uuid4
from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

MAX_STEP = 10
STEP_ITEMS = 5

class PipeAbortException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def get_crop_box(width:int, height:int, crop_width:int, crop_height:int, horizontal_offset: int = 0.5, vertical_offset: int = 0.5):
        left = (width - crop_width) * horizontal_offset
        top = (height - crop_height) * vertical_offset
        right = crop_width + (width - crop_width) * horizontal_offset
        botton = crop_height + (height - crop_height) * vertical_offset

        return (left, top, right, botton)

def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

class ImageAugmentationActions():
    _actions = []
    _actions_dict = {}
    
    @classmethod
    def add_action(cls, func, str_id, desc):
        t = {
            "func": func,
            "desc": desc,
            "id": str_id,
        }
        cls._actions.append(t)
        cls._actions_dict[str_id] = t

def augmentation_action(id: str, desc: str):
    def decorator(func):
        ImageAugmentationActions.add_action(func, id, desc)
        return func
    return decorator

class ImageAugmentationPipeline():
    def __init__(self, pipe_cfg, indir, outdir, outdir_with_step) -> None:
        self.indir = indir
        self.outdir = outdir
        self.pipe_cfg = pipe_cfg
        
        self.crop_pixel_min = int(pipe_cfg.global_opt.crop_pixel_min)
        self.crop_scale_min = pipe_cfg.global_opt.crop_scale_min
        self.crop_scale_max = pipe_cfg.global_opt.crop_scale_max
        self.outdir_with_step = outdir_with_step
        self.outputs_image = []
        self.models = {}
        self.step_dir = {}

        self.action_info = []
        self.init_action()

    def init_action(self):
        for act in self.pipe_cfg.actions:
            self.action_info.append({
                "cfg": act,
                "attributes": set(act.attributes.split(" "))
            })

    def process_image(self, image_filename):
        image_filename = os.path.abspath(image_filename)

        self.output_prefix = os.path.join(self.outdir, str(uuid4()))

        if self.outdir_with_step:
            for i in range(1, MAX_STEP + 1):
                self.step_dir[i] = os.path.join(self.outdir, f"{i}")
                if not os.path.isdir(self.step_dir[i]):
                    os.mkdir(self.step_dir[i])

        self.outputs_image.append(Image.open(image_filename))
        i = 0
        try:
            for act_info in self.action_info:
                self.process_step(i, act_info) 
                i += 1
        except PipeAbortException:
            pass
        finally:
            for v in self.outputs_image:
                v.close()
            self.outputs_image = []

    def process_step(self, index, act_info):
        act_cfg, attributes = act_info["cfg"], act_info["attributes"]

        try:
            t = ImageAugmentationActions._actions_dict[act_cfg.id]
            func = t["func"]
        except AttributeError as e:
            raise gr.Error(f"无效的生成操作")

        if "input" in attributes:
            input_image = self.outputs_image[-1]
        else:
            input_image = self.outputs_image[0]
        
        output_image: Image = func(self, input_image)

        if ("continue" not in attributes) and (output_image is None):
            raise PipeAbortException() 
        
        if "save" in attributes and output_image:
            if self.outdir_with_step:
                output_image.save(os.path.join(self.step_dir[index + 1], f"{uuid4()}_step_{index + 1}.png"), "PNG")
            else:
                output_image.save(os.path.join(self.outdir, f"{uuid4()}_step_{index + 1}.png"), "PNG")
            if "output" in attributes:
                if len(self.outputs_image) > 1: 
                    self.outputs_image[-1].close()
                    self.outputs_image[-1] = output_image
                else:
                    self.outputs_image.append(output_image)
            else:
                output_image.close()
                del output_image

    @augmentation_action(id="abort", desc="中止")
    def _act_0(self, input_image: Image):
        raise PipeAbortException()

    @augmentation_action(id="flip_left_right", desc="水平翻转")
    def _act_1(self, input_image: Image):
        """水平翻转"""
        out = input_image.transpose(Image.FLIP_LEFT_RIGHT)
        return out

    @augmentation_action(id="flip_top_bottom", desc="垂直翻转")
    def _act_2(self, input_image: Image):
        """垂直翻转"""
        out = input_image.transpose(Image.FLIP_TOP_BOTTOM)
        return out

    @augmentation_action(id="random_crop", desc="随机裁剪 - 原始比例")
    def _act_3(self, input_image: Image):
        """随机裁剪 - 原始比例"""
        w, h = input_image.size
        scale = 1
        
        if self.crop_scale_min >= self.crop_scale_max:
            scale = self.crop_scale_min
        else:
            scale = random.randint(self.crop_scale_min, self.crop_scale_max)
        new_w, new_h = math.floor(w * scale), math.floor(h * scale)

        if new_w < self.crop_pixel_min or new_h < self.crop_pixel_min:
            return None

        crop_box = get_crop_box(w, h, new_w, new_h, random.uniform(0, 1), random.uniform(0, 1))

        image = input_image.crop(crop_box)
        out = image.resize((new_w, new_h))
        return out

    @augmentation_action(id="random_crop - 1:1", desc="随机裁剪 - 1:1比例")
    def _act_4(self, input_image: Image):
        """随机裁剪 - 1:1比例"""
        w, h = input_image.size
        scale = 1
        
        if self.crop_scale_min >= self.crop_scale_max:
            scale = self.crop_scale_min
        else:
            scale = random.randint(self.crop_scale_min, self.crop_scale_max)

        new_edge = math.floor(min(w, h) * scale)

        if new_edge < self.crop_pixel_min:
            return None

        crop_box = get_crop_box(w, h, new_edge, new_edge, random.uniform(0, 1), random.uniform(0, 1))

        image = input_image.crop(crop_box)
        out = image.resize((new_edge, new_edge))
        return out

    @augmentation_action(id="YOLOS", desc="YOLOS - 主体标签")
    def _act_5(self, input_image: Image):
        """YOLOS - 主体标签"""
        yolos_model = get_yolos_model()
        with yolos_model.get_model() as m:
            inputs = m.processor(input_image, return_tensors="pt").to(yolos_model.device)
            outputs = m.model(**inputs)
            logits = outputs.logits
            bboxes = outputs.pred_boxes

        target_sizes = torch.tensor([input_image.size[::-1]])
        results = m.processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

        max_pixel = None
        max_box = None
        max_size = None

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            size=(int(box[2]-box[0]), (int)(box[3]-box[1]))

            if size[0] < self.crop_pixel_min or size[1] < self.crop_pixel_min:
                continue

            if max_pixel is None or max_pixel < (size[0] * size[1]):
                max_pixel = (size[0] * size[1])
                max_box = box
                max_size = size
        
        if max_box is None:
            return None

        image = input_image.crop(max_box)
        image = image.resize(max_size)

        return image

def hf_download(repo_id: str, file: str):
    try:
        kwargs = {}
        if get_webui_configs().get_cfg().model.offline_load:
            kwargs["local_files_only"] = True
            kwargs["force_download"] = False
        path = hf_hub_download(repo_id, file, **kwargs)
    except Exception:
        msg = f"Failed to load model {file!r} from huggingface"
        print(msg)
        path = "INVALID"
    return path

def create_yolo_action(model_id):
    def load_yolo_model(obj: ImageAugmentationPipeline):
        if model_id in obj.models:
            return obj.models[model_id]
        yolo_model_list = get_yolo_model_list()
        model_cfg = yolo_model_list.get_model(model_id)
        path = hf_download(model_cfg.repo_id, model_cfg.file)
        model = YOLO(path)
        obj.models[model_id] = model
        return model
    def on_yolo(obj: ImageAugmentationPipeline, input_image: Image):
        model = load_yolo_model(obj)
        confidence=0.65
        pred = model(input_image, conf=confidence, device=torch.device("cuda"))
        bboxes = pred[0].boxes.xyxy.cpu().numpy()
        if bboxes.size == 0:
            return None
        bboxes = bboxes.tolist()

        max_pixel = None
        max_box = None
        max_size = None

        for box in bboxes:
            size=(int(box[2]-box[0]), (int)(box[3]-box[1]))
            if size[0] < obj.crop_pixel_min or size[1] < obj.crop_pixel_min:
                continue

            if max_pixel is None or max_pixel < (size[0] * size[1]):
                max_pixel = (size[0] * size[1])
                max_box = box
                max_size = size
        
        if max_box is None:
            return None

        image = input_image.crop(max_box)
        image = image.resize(max_size)

        return image
    return on_yolo

def on_to_yaml_config(crop_pixel_min, crop_scale_min, crop_scale_max, *args):
    global_opt = DictConfig({
        "crop_pixel_min": int(crop_pixel_min),
        "crop_scale_min": crop_scale_min,
        "crop_scale_max": crop_scale_max,
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
            "extra_params": "",
            "attributes": " ".join(attributes)
        })

        actions.append(act)

    cfg = DictConfig({
        "global_opt": global_opt,
        "actions": ListConfig(actions)
    })

    text = OmegaConf.to_yaml(cfg)
    return text

def on_process_yaml_pipeline(indir, outdir, outdir_with_step, yaml_str, progress = gr.Progress(track_tqdm=True)):
    if not indir or not os.path.exists(indir):
        raise gr.Error(f"输入目录不存在: {indir}")
    if not outdir:
        raise gr.Error(f"无效的输出目录: {outdir}")

    cfg = OmegaConf.create(yaml_str)

    ds = SrcDataset(indir)

    if len(ds) == 0:
        raise gr.Error(f"指定输入目录下无可用图片")

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    pipe = ImageAugmentationPipeline(cfg, indir, outdir, outdir_with_step)

    for image in tqdm(ds):
        pipe.process_image(image)
        
    del pipe

    return "完成"

def create_image_augmentation(top_elems: TopElements):
    PRESETS=["中止", "水平翻转", "垂直翻转", "随机裁剪 - 原始比例", "随机裁剪 - 1:1比例", "YOLOS - 最大像素盒"]

    yolo_model_list = get_yolo_model_list()

    #for i in  [ for i in yolo_model_list.get_cfg()]
    for i in yolo_model_list.get_cfg():
        yolo_id =  f"YOLO - {i.model_id}"
        ImageAugmentationActions.add_action(create_yolo_action(i.model_id), yolo_id, yolo_id)

    drop_desc = [i["desc"] for i in ImageAugmentationActions._actions]

    with gr.Row():
        indir = gr.Textbox(label="输入目录")
        default_save_path = os.path.join(Path.home(), "Pictures", "CLIPImageSearchWebUI", "Augmentation")
        outdir = gr.Textbox(label="输出目录", value=default_save_path)
        outdir_with_step = gr.Checkbox(label="输出至子目录中", info="为每个步骤分别建立子级目录进行输出", value=True, interactive=True)

    with gr.Tabs() as tabs:
        with gr.TabItem(label="普通模式", id=0):
            with gr.Row():
                convert_to_yaml_cfg = gr.Button("转换到配置")
                generate_btn = gr.Button("生成", variant="primary")

            with gr.Row():
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
                generate_from_yaml_btn = gr.Button("从配置生成", variant="primary")
            pipline_yaml = gr.Code(label="Pipline YAML", language="yaml", interactive=True)

    internal_pipline_yaml = gr.State()

    convert_to_yaml_cfg.click(on_to_yaml_config, [crop_pixel_min, crop_scale_min, crop_scale_max] + components, [pipline_yaml]).then(lambda : gr.Tabs.update(selected=1), [], [tabs])
    generate_btn.click(on_to_yaml_config, [crop_pixel_min, crop_scale_min, crop_scale_max] + components, [internal_pipline_yaml]).then(on_process_yaml_pipeline, [indir, outdir, outdir_with_step, internal_pipline_yaml], [top_elems.msg_text])
    generate_from_yaml_btn.click(on_process_yaml_pipeline, [indir, outdir, outdir_with_step, pipline_yaml], [top_elems.msg_text])

    

    