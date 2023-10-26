import os
import gradio as gr
from tqdm import tqdm
from module.foundation.webui import TopElements
from module.data import get_cache_root, get_webui_configs
from module.core.src_datasets import SrcDataset
from PIL import Image
from uuid import uuid4
from pathlib import Path

PIPELINE=["中止", "水平翻转", "垂直翻转", "随机裁剪 - 原始比例", "随机裁剪 - 1:1比例", "YOLO-TOP1-标签", "YOLO-随机标签"]

class PipeAbortException(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def divide_chunks(l, n): 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 

class ImageAugmentationPipeline():
    def __init__(self, indir, outdir, crop_pixel_min, crop_scale_min, crop_scale_max) -> None:
        self.indir = indir
        self.outdir = outdir
        self.crop_pixel_min = crop_pixel_min
        self.crop_scale_min = crop_scale_min
        self.crop_scale_max = crop_scale_max
        self.outputs_image = []

    def process_image(self, image_filename, steps_args):
        image_filename = os.path.abspath(image_filename)

        self.output_prefix = os.path.join(self.outdir, str(uuid4()))

        self.outputs_image.append(Image.open(image_filename))
        i = 0

        try:
            for arg in divide_chunks(steps_args, 4):
                self.process_step(i, *arg)
                i += 1
        except PipeAbortException:
            pass
        finally:
            for v in self.outputs_image:
                v.close()
            self.outputs_image = []

    def process_step(self, index, pipe_action, get_input, send_output, save_file):
        try:
            func = getattr(self, f"_act_{pipe_action}")
        except AttributeError as e:
            raise gr.Error(f"无效的生成操作")

        if get_input:
            input_image = self.outputs_image[-1]
        else:
            input_image = self.outputs_image[0]
        
        output_image: Image = func(input_image)
        
        if save_file and output_image:
            output_image.save(self.output_prefix + f"_step_{index}.png","PNG")
            if send_output:
                if len(self.outputs_image) > 1: 
                    self.outputs_image[-1].close()
                    self.outputs_image[-1] = output_image
                else:
                    self.outputs_image.append(output_image)
            else:
                output_image.close()
                del output_image

    def _act_0(self, input_image: Image):
        raise PipeAbortException()

    def _act_1(self, input_image: Image):
        """水平翻转"""
        out = input_image.transpose(Image.FLIP_LEFT_RIGHT)
        return out

    def _act_2(self, input_image: Image):
        """垂直翻转"""
        out = input_image.transpose(Image.FLIP_TOP_BOTTOM)
        return out

def on_process_data_augmentation_pipeline(indir, outdir, crop_pixel_min, crop_scale_min, crop_scale_max, *args, progress = gr.Progress(track_tqdm=True)):
    if not indir or not os.path.exists(indir):
        raise ValueError(f"输入目录不存在: {indir}")
    if not outdir:
        raise ValueError(f"无效的输出目录: {outdir}")
    
    ds = SrcDataset(indir)

    if len(ds) == 0:
        raise ValueError(f"指定输入目录下无可用图片")

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    pipe = ImageAugmentationPipeline(indir, outdir, crop_pixel_min, crop_scale_min, crop_scale_max)

    for image in tqdm(ds):
        pipe.process_image(image, args)
        
    del pipe

    return "完成"

def create_image_augmentation(top_elems: TopElements):
    with gr.Row():
        indir = gr.Textbox(label="输入目录")
        default_save_path = os.path.join(Path.home(), "Pictures", "CLIPImageSearchWebUI", "Augmentation")
        outdir = gr.Textbox(label="输出目录", value=default_save_path)

    generate_btn = gr.Button("生成", variant="primary")

    with gr.Row():
        crop_pixel_min = gr.Number(label="裁剪最小边", value=512, info="裁剪结果当最小边小于此值时，将丢弃输出", interactive=True)
        crop_scale_min = gr.Slider(label="裁剪起始比例", value=0.5, minimum=0.01, maximum=1.0, step=0.01, interactive=True)
        crop_scale_max = gr.Slider(label="裁剪结束比例", value=0.5, minimum=0.01, maximum=1.0, step=0.01, interactive=True)

    components = []

    for i in range(1, 11):
        with gr.Row():
            pipe_action = gr.Dropdown(label=f"步骤 {i}", choices=PIPELINE, scale=5, type="index", value=0)
            get_input = gr.Checkbox(label="接收输入", value=True, interactive=True)
            send_output = gr.Checkbox(label="发送输出", value=True, interactive=True)
            save_file = gr.Checkbox(label="保存文件", value=True, interactive=True)

            components += [pipe_action, get_input, send_output, save_file]

    generate_btn.click(on_process_data_augmentation_pipeline, [indir, outdir, crop_pixel_min, crop_scale_min, crop_scale_max] + components, [top_elems.msg_text])

    

    