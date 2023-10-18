import torch
import gradio as gr
import os
import sys
import time
from tqdm import tqdm
from module.core.src_datasets import SrcDataset
from PIL import Image, ImageFile
from module.data import get_clip_model, get_webui_configs
import shutil
import module.utils.run_util as run_util

from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import FakeEmbeddings
from module.foundation.webui import TopElements

ImageFile.LOAD_TRUNCATED_IMAGES = True

def on_create_embeddings(indirs, batch_size, num_workers, progress=gr.Progress(track_tqdm=True)):
    clip_model = get_clip_model()

    create_dirs = []

    for dir in indirs.split("\n"):
        dir = dir.strip()
        if len(dir) == 0:
            continue
        if not os.path.isdir(dir):
            raise gr.Error(f"不是合法的路径: {dir}")
        create_dirs.append(dir)

    t0 = time.time()

    args = [ 
        'scripts/create_embeddings.py',
        '--batch_size', int(batch_size),
        '--num_workers', int(num_workers),
        '--clip_model_id', f'"{clip_model.clip_model_id}"'
    ]

    for dir in create_dirs:
        args.append('--indir')
        args.append(f'"{dir}"')
    
    run_cmd = sys.executable + " " + " ".join([str(i) for i in args] )
    
    clip_model.release_model()
    
    print("执行脚本: " + run_cmd)
    
    run_util.run(run_cmd) 
    
    t1 = time.time()

    return f"操作完成, 耗时: {t1-t0}"

def build_vector_db(indir, batch_size, is_rebuild):
    batch_size = int(batch_size)

    if not indir:
        raise gr.Error("输入目录不能为空")

    indir = os.path.abspath(indir)

    if indir[-1] == '\\' or indir[-1] == '/':
        indir = indir[:-1]

    prefix_len = len(indir) + 1

    clip_model = get_clip_model()

    save_db_dir = os.path.join(indir, ".vec_db", clip_model.get_model_subpath())

    if os.path.exists(save_db_dir):
        if is_rebuild:
            shutil.rmtree(save_db_dir)
        else:
            raise gr.Error(f"目录已存在, 如果需要再次构建请手动删除：{save_db_dir}")

    os.makedirs(save_db_dir)

    print(f"准备开始构建: {indir} batch_size={batch_size}")

    embed_file_tail=f".{clip_model.get_short_name()}.embed.pt"
    embed_file_tail_len = len(embed_file_tail)

    ds = SrcDataset(root=indir, endswith_str=embed_file_tail)

    print(f"向量数量: {len(ds)}")

    if len(ds) == 0:
        msg = f"指定目录下无任何有效图片元信息: {indir}"
        print(msg)
        raise gr.Error(msg)

    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)

    vectorstore = None

    #embedding_list = []
    #file_list = []

    torch_emb_index = 0
    dim = clip_model.get_dim()

    embed_size = torch.Size([dim])

    device = torch.device('cpu')

    for batch in tqdm(loader):
        data = []

        for i, embed_filename in enumerate(batch):
            try:
                embedding = torch.load(embed_filename, map_location=device)
            except Exception as e:
                print(f"read embedding file error: {embed_filename}")
                continue
            
            if len(embedding) != dim:
                raise gr.Error("图片的向量文件与当前选择的CLIP模型维度不匹配")

            embedding /= embedding.norm(dim=-1, keepdim=True) 

            image_name = embed_filename[:-embed_file_tail_len]
            image_name = image_name[prefix_len:]
            data.append((image_name, embedding.tolist()))
            del embedding

        vectorstore_new = FAISS.from_embeddings(text_embeddings=data, embedding=FakeEmbeddings(size=len(data)))

        torch_emb_index+=1
        
        if vectorstore is None:
            vectorstore = vectorstore_new
        else:
            vectorstore.merge_from(vectorstore_new)

    vectorstore.save_local(save_db_dir)
    del vectorstore
    #embedding_slice = torch.cat(embedding_list)

    #torch.save(embedding_slice, f"{save_db_dir}\\embedding.pt")

    #with open(f"{save_db_dir}\\files.json", "w") as f:
        #json.dump(file_list, f)

    del data
    return f"完成: {indir}, 向量维度: {embed_size}"

def on_build_vector_db(indirs, batch_size, is_rebuild, progress=gr.Progress(track_tqdm=True)):
    create_dirs = []

    for dir in indirs.split("\n"):
        dir = dir.strip()
        if len(dir) == 0:
            continue
        if not os.path.isdir(dir):
            raise gr.Error(f"不是合法的路径: {dir}")
        create_dirs.append(dir)

    for dir in tqdm(create_dirs):
        build_vector_db(dir, batch_size, is_rebuild)

def on_copy_vec_db_dir_from_configs():
    lines = []
    for path in get_webui_configs().get_cfg().librarys:
        if path.startswith("<"):
            end_pos = path.find(">")
            if end_pos >= 0:
                path = path[end_pos+1:]
        lines.append(path)
    
    text = "\n".join(lines)

    return text, text

def create_embedding_page(top_elems: TopElements):
    gr.Markdown(value="""在加载指定目录的数据集的向量库到内存之前，必须先生成所有图片的 embedding 向量文件，再构建向量库缓存(可选)。""")
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("(第一步) 生成 embedding 向量文件")
        copy_vec_db_dir_from_configs = gr.Button("从设置中复制所有向量库路径")
    build_meta_indir = gr.TextArea(label="生成此目录列表下所有图片向量文件 (需要较高的 CPU 和 GPU 资源)")
    with gr.Row():
        build_meta_batch_size = gr.Number(label="batch_size", info="批大小", value=4)
        build_meta_num_workers = gr.Number(label="num_workers", info="工作进程数", value=8)
        start_create_meta_btn = gr.Button(value="生成向量文件")

    gr.Markdown("(第二步) 生成向量库磁盘文件，如果不生成则会在每次加载时遍历所有向量文件在内存中构建。")
    build_db_indir = gr.TextArea(label="生成此目录列表下所有向量文件的向量库，并存储到各目录中")
    with gr.Row():
        build_db_batch_size = gr.Number(label="batch_size", info="批大小", value=5000)
        rebuild_checkbox = gr.Checkbox(label="强制重建", info="已存在向量库缓存时，删除旧的", value=False)
        start_create_vector_index_btn = gr.Button(value="构建目录向量索引库")

    start_create_meta_btn.click(fn=on_create_embeddings, inputs=[build_meta_indir, build_meta_batch_size, build_meta_num_workers], outputs=[top_elems.msg_text])
    start_create_vector_index_btn.click(fn=on_build_vector_db, inputs=[build_db_indir, build_db_batch_size, rebuild_checkbox], outputs=[top_elems.msg_text])

    copy_vec_db_dir_from_configs.click(on_copy_vec_db_dir_from_configs, [], [build_meta_indir, build_db_indir])