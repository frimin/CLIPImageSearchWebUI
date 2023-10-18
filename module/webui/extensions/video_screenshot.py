import gradio as gr
import torch
from module.core.src_datasets import SrcDataset
from tqdm import tqdm
import os
import shutil
import module.utils.run_util as run_util
import time
import json
from module.foundation.webui import TopElements

# Get-ChildItem -Include '*(__screenshot__)*' -Recurse -force | Remove-Item -Force -Recurse

def on_start_process(indir: str, file_exts: str, screenshot_speed: float, ignore_ffmpeg_err: bool, gpu_enable: bool, progress=gr.Progress()):
    exts = [i.strip() for i in file_exts.split(',')]

    if len(exts) == 0:
        raise gr.Error("需要指定文件扩展名")

    if screenshot_speed <= 0:
        raise gr.Error("截图速率必须大于 0")

    if not indir:
        raise gr.Error("需要指定一个输入目录")

    # 检查 ffmpeg 可用
    try:
        run_util.run("ffmpeg -version")
    except Exception:
        raise gr.Error("ffmpeg 不可用，确认其正确安装且在 PATH 环境变量中")

    ds = SrcDataset(root=indir, ext=exts)

    t0 = time.time()

    loader = torch.utils.data.DataLoader(ds, batch_size=1)

    for batch in progress.tqdm(tqdm(loader)):
        video_name, _ = os.path.splitext(batch[0])

        screenshot_dir = video_name + " (__screenshot__)"
        screenshot_meta_file = os.path.join(screenshot_dir, '.screenshot_meta.json')

        if os.path.exists(screenshot_meta_file):
            continue

        if os.path.exists(screenshot_dir):
            shutil.rmtree(screenshot_dir)

        os.mkdir(screenshot_dir)

        try:
            gpu_arg = ""
            if gpu_enable:
                gpu_arg = "-hwaccel cuda"
            run_util.run(f'ffmpeg {gpu_arg}  -i "{batch[0]}" -r 1/{screenshot_speed} -q:v 1 "{screenshot_dir}/%08d.jpg"')
            #run_util.run(f'ffmpeg -hwaccel cuda -i "{batch[0]}" -r 1/1 -qmin 1 "{screenshot_dir}/%03d.jpg"')
        except Exception as e:
            if ignore_ffmpeg_err:
                print(e)
            else:
                raise gr.Error("调用 ffmpeg 遇到错误")

        # 有文件输出才标记完成 
        if len(os.listdir(screenshot_dir)) > 0:
            with open(screenshot_meta_file, "w") as f:
                meta = { 'speed': screenshot_speed }
                json.dump(meta, f)
        else:
            #os.remove(screenshot_dir)
            pass

    t1 = time.time()

    return f"完成, 耗时 {t1-t0}"

def create_video_screenshot(top_elems: TopElements):
    gr.Markdown(value="""通过 [ffmpeg](https://www.ffmpeg.org/) 给视频以指定的频率截图，再构建图片向量库来做到搜索视频内容""")

    indir = gr.Textbox(label="处理的目录", value="")
    with gr.Row():
        file_exts = gr.Textbox(label="视频扩展名", value=".mkv,.mp4,.webm")
        screenshot_speed = gr.Number(label="每N秒截图一张", value=1)
        ignore_ffmpeg_err = gr.Checkbox(label="忽略 ffmpeg 错误", value=True)
        gpu_enable = gr.Checkbox(label="使用GPU(不推荐，速度没有明显区别)", value=False)
        start_screenshot = gr.Button("批量视频截图", variant="primary")

    start_screenshot.click(fn=on_start_process, inputs=[indir, file_exts, screenshot_speed, ignore_ffmpeg_err, gpu_enable], outputs=[top_elems.msg_text])