import os
import gradio as gr
import random
import math
import torch
from tqdm import tqdm
from omegaconf import OmegaConf, ListConfig, DictConfig
from module.core.src_datasets import SrcDataset
from PIL import Image
from uuid import uuid4
from pathlib import Path
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

MAX_STEP = 10
STEP_ITEMS = 5

def get_crop_box(width:int, height:int, crop_width:int, crop_height:int, horizontal_offset: float = 0.5, vertical_offset: float = 0.5, crop_overlap: float = 0):
        horizontal_offset = min(1, max(horizontal_offset, 0))
        vertical_offset = min(1, max(vertical_offset, 0))
        crop_overlap = min(1, max(crop_overlap, 0))

        overlap_width = crop_width * crop_overlap / 2.0
        overlap_height = crop_height * crop_overlap / 2.0

        left = (width - crop_width) * horizontal_offset - overlap_width
        top = (height - crop_height) * vertical_offset - overlap_height
        right = crop_width + (width - crop_width) * horizontal_offset + overlap_width
        botton = crop_height + (height - crop_height) * vertical_offset + overlap_height

        left = min(width, max(left, 0))
        top = min(height, max(top, 0))
        right = min(width, max(right, 0))
        botton = min(height, max(botton, 0))

        box = (int(left), int(top), int(right), int(botton))
        size = (box[2] - box[0], box[3] - box[1])

        return box, size 

def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

class PresetsList():
    value = None

    @classmethod
    def init(cls):
        if cls.value is None:
            presets_dir = "./data/presets/image_augmentation"
            cls.value = []
            for file in os.listdir(presets_dir):
                name, ext = os.path.splitext(file)
                if ext != ".yaml":
                    continue
                cls.value.append((name, os.path.join(presets_dir, file)))

class PipeAbortException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

class ImageAugmentationActions():
    _actions = []
    _actions_dict = {}

    @classmethod
    def init(cls, yolo_model_list):
        for i in yolo_model_list.get_cfg():
            yolo_id =  f"YOLO - {i.model_id}"
            cls.add_action(create_yolo_action(i.model_id, yolo_model_list), yolo_id, yolo_id)
    
    @classmethod
    def add_action(cls, func, str_id, desc, visible=True):
        t = {
            "func": func,
            "desc": desc,
            "id": str_id,
            "visible": visible,
        }
        cls._actions.append(t)
        cls._actions_dict[str_id] = t

def augmentation_action(id: str, desc: str, visible=True):
    def decorator(func):
        ImageAugmentationActions.add_action(func, id, desc, visible)
        return func
    return decorator

class ImageAugmentationPipeline():
    def __init__(self, pipe_cfg, indir, outdir, outdir_with_step) -> None:
        self.indir = indir
        self.outdir = outdir
        self.pipe_cfg = pipe_cfg

        global_opt = DictConfig({
            "crop_pixel_min": int(pipe_cfg.opt.crop_pixel_min),
            "crop_scale_min": pipe_cfg.opt.crop_scale_min,
            "crop_scale_max": pipe_cfg.opt.crop_scale_max,
            "skip_pixel": int(pipe_cfg.opt.skip_pixel)
        })

        self.default_action_param = DictConfig({
            "opt": global_opt
        })

        self.outdir_with_step = outdir_with_step
        self.outputs_image = []
        self.models = {}
        self.step_dir = {}

        self.action_info = []
        self._init_action()
        self._mkdirs()

    def _init_action(self):
        for act in self.pipe_cfg.actions:
            params = act.get("params")

            if not params:
                params = OmegaConf.create()

            assert isinstance(params, DictConfig)

            params = OmegaConf.merge(self.default_action_param, params)

            self.action_info.append({
                "cfg": act,
                "attributes": set(act.attributes.split(" ")),
                "params": params,
            })

    def _mkdirs(self):
        if self.outdir_with_step:
            for i in range(1, MAX_STEP + 1):
                self.step_dir[i] = os.path.join(self.outdir, f"{i}")
                if not os.path.isdir(self.step_dir[i]):
                    os.mkdir(self.step_dir[i])

    def _valid_size(self, w, h, act_params):
        if w < act_params.opt.crop_pixel_min or h < act_params.opt.crop_pixel_min:
            return False
        return True

    def process_image(self, image_filename):
        image_filename = os.path.abspath(image_filename)

        img = Image.open(image_filename)
        size = img.size

        if size[0] < self.default_action_param.opt.skip_pixel or size[1] < self.default_action_param.opt.skip_pixel:
            del img
            return

        self.outputs_image.append(img)
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
        act_cfg, attributes, act_params = act_info["cfg"], act_info["attributes"], act_info["params"]

        try:
            t = ImageAugmentationActions._actions_dict[act_cfg.id]
            func = t["func"]
        except AttributeError as e:
            raise gr.Error(f"无效的生成操作")

        if "input" in attributes:
            input_image = self.outputs_image[-1]
        else:
            input_image = self.outputs_image[0]
        
        output_image: Image = func(self, input_image, act_params)

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
    def act_abort(self, input_image: Image, act_params):
        raise PipeAbortException()

    @augmentation_action(id="flip_left_right", desc="水平翻转")
    def act_flip_left_right(self, input_image: Image, act_params):
        """水平翻转"""
        out = input_image.transpose(Image.FLIP_LEFT_RIGHT)
        return out

    @augmentation_action(id="flip_top_bottom", desc="垂直翻转")
    def act_flip_top_bottom(self, input_image: Image, act_params):
        """垂直翻转"""
        out = input_image.transpose(Image.FLIP_TOP_BOTTOM)
        return out

    @augmentation_action(id="random_crop", desc="随机裁剪 - 原始比例")
    def act_random_crop(self, input_image: Image, act_params):
        """随机裁剪 - 原始比例"""
        w, h = input_image.size
        scale = 1
        
        if act_params.opt.crop_scale_min >= act_params.opt.crop_scale_max:
            scale = act_params.opt.crop_scale_min
        else:
            scale = random.randint(act_params.opt.crop_scale_min, act_params.opt.crop_scale_max)
        new_w, new_h = math.floor(w * scale), math.floor(h * scale)

        if not self._valid_size(new_h, new_w, act_params):
            return None

        x_offset = act_params.get("x")
        y_offset = act_params.get("y")

        if x_offset is None:
            x_offset = random.uniform(0, 1)
        if y_offset is None:
            y_offset = random.uniform(0, 1)

        crop_box, new_size = get_crop_box(w, h, new_w, new_h, x_offset, y_offset, act_params.get("overlap", 0))

        image = input_image.crop(crop_box)
        out = image.resize(new_size)
        return out

    @augmentation_action(id="random_crop - square", desc="随机裁剪 - 1:1比例")
    def act_random_crop_square(self, input_image: Image, act_params):
        """随机裁剪 - 1:1比例"""
        w, h = input_image.size
        scale = 1

        if act_params.opt.crop_scale_min >= act_params.opt.crop_scale_max:
            scale = self.crop_scale_min
        else:
            scale = random.randint(act_params.opt.crop_scale_min, act_params.opt.crop_scale_max)

        new_edge = math.floor(min(w, h) * scale)

        x_offset = act_params.get("x")
        y_offset = act_params.get("y")

        if x_offset is None:
            x_offset = random.uniform(0, 1)
        if y_offset is None:
            y_offset = random.uniform(0, 1)

        crop_box, new_size = get_crop_box(w, h, new_edge, new_edge, x_offset, y_offset, act_params.get("overlap", 0))

        image = input_image.crop(crop_box)
        out = image.resize(new_size)
        return out
    @augmentation_action(id="resize", desc="缩放")
    def act_resize(self, input_image: Image, act_params):
        resize_to = act_params.opt.crop_pixel_min

        w, h = input_image.size

        if w > h:
            scale = resize_to / h 
        else:
            scale = resize_to / w
        
        new_w, new_h = int(w * scale), int(h * scale)

        out = input_image.resize((new_w, new_h)) 
        return out

    """
    @augmentation_action(id="YOLOS", desc="YOLOS - 主体标签")
    def act_yolos(self, input_image: Image, act_params):
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

            if not self._valid_size(size[0], size[1]):
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
    """

def hf_download(repo_id: str, file: str):
    try:
        kwargs = {}
        #if get_webui_configs().get_cfg().model.offline_load:
        #    kwargs["local_files_only"] = True
        #    kwargs["force_download"] = False
        path = hf_hub_download(repo_id, file, **kwargs)
    except Exception:
        msg = f"Failed to load model {file!r} from huggingface"
        print(msg)
        path = "INVALID"
    return path

def create_yolo_action(model_id, yolo_model_list):
    def load_yolo_model(obj: ImageAugmentationPipeline):
        if model_id in obj.models:
            return obj.models[model_id]
        model_cfg = yolo_model_list.get_model(model_id)
        path = hf_download(model_cfg.repo_id, model_cfg.file)
        model = YOLO(path)
        obj.models[model_id] = model
        return model
    def on_yolo(obj: ImageAugmentationPipeline, input_image: Image, act_params):
        model = load_yolo_model(obj)
        confidence=0.7
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
            if not obj._valid_size(size[0], size[1], act_params):
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