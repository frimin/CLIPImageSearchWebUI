import os
import gradio as gr
from tqdm import tqdm
from module.utils.constants_util import DISABLE_DELETE_FILE_ERROR
from module.foundation.webui import TopElements
from module.data import get_cache_root, get_webui_configs
from module.core.src_datasets import SrcDataset

def on_check_file(indir, file_exts, file_size, progress=gr.Progress(track_tqdm=True)):
    if len(indir) == 0 or not os.path.isdir(indir):
        raise gr.Error("无效的路径")
    file_exts = [i.strip() for i in file_exts.split(",")]
    ds = SrcDataset(root=indir, ext=file_exts) 
    temp_file = get_cache_root().create_temporary_filename(".txt")

    with open(temp_file, "w") as f:
        for i in tqdm(ds, desc="检查文件"):
            stats = os.stat(i)
            if stats.st_size < file_size:
                f.write(i + "\n")

    return temp_file, "检查完毕"

def on_delete_file(file, progress=gr.Progress(track_tqdm=True)):
    if get_webui_configs().get_cfg().security.disalbe_delete_file:
        raise DISABLE_DELETE_FILE_ERROR 

    if file is None:
        return "无文件"

    delete_n = 0

    with open(file.name) as f:
        files = f.readlines()
        for filename in tqdm(files):
            filename = filename.strip()
            if os.path.exists(filename):
                os.remove(filename)
                delete_n += 1
    return f"已删除 {delete_n} 项" 

def create_file_checker(top_elems: TopElements):
    gr.Markdown(value="""检查指定目录过小的文件并输出一个路径列表文件""")
    with gr.Row():
        indir = gr.Textbox(label="检查目录", info="此目录下的所有子目录将会被检查", scale=3)
        file_exts = gr.Textbox(label="扩展名", info="扩展名", value=".png,.jpg,.jpeg,.webp,.pt")
        file_size = gr.Number(label="文件大小", info="单位字节,小于此大小的文件将被输出路径", value=1024)
        check_btn = gr.Button(value="开始检查 (无删除)")
    file = gr.File(label="列表文件", visible=True, file_count="single")

    check_btn.click(on_check_file, [indir, file_exts, file_size], [file, top_elems.msg_text])

    delete_btn = gr.Button(value="从列表文件删除所有指定路径的文件 (不可恢复)")

    delete_btn.click(on_delete_file, [file], [top_elems.msg_text])