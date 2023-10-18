import gradio as gr
from module.search_vector_core.search_state import SearchVectorPageState
import module.utils.constants_util as constants_util
import faiss
import torch
import time
import uuid
from tqdm import tqdm
import os

import gradio as gr
from module.search_vector_core.search_state import SearchVectorPageState
from module.data import get_clip_model, get_vector_db_mgr
import module.utils.constants_util as constants_util
import faiss
import torch
import time
import uuid
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from datasets import Dataset
from .aesthetic_predictor_mlp import MLP
from sklearn.cluster import DBSCAN

model = None

def on_gui():
    with gr.Tab(label="分块去重"):
        gr.Markdown("基于美学评分结果排序的分块聚类去重: 由于重复的图片美学评分也是相近的，所有基于美学评分的结果排序后进行分块，再对每个块进行低阈值的 DBSCAN 聚类，则可以一定程度上排除重复图片。")
        with gr.Row():
            chunk_size = gr.Number(label="分块大小", info="执行聚类算法的块大小，越大的块在更大的范围内消除重复图片，但是计算很慢。 [1000,50000]", value=10000, minimum=1000, maximum=50000)
            threshold = gr.Slider(label="DBSCAN 聚类阈值", info="较小的阈值让更相似的图片分为一组， 较大的阈值让更不相似的图片分为一组。去重后的结果将从组中选择一张图片。[0.05,1]" , 
                                  value=0.2, minimum=0.05, maximum=1, step=0.05)

            choices = ["排除的重复图片以及对应保留图", "仅输出排除的重复图片"]

            export_duplicate_pair = gr.Dropdown(label="附加结果输出", choices=choices, info="选择仅输出排除的重复图片，方便批量删除重复的内容", type="index", value=choices[0], interactive=True)

        btn = gr.Button(value="分块去重", variant="primary")

    return [chunk_size, threshold, export_duplicate_pair, btn]

def on_bind(search_state: SearchVectorPageState, compolents: list[gr.components.Component]):
    def on_deduplication(chunk_size: float, 
                        threshold: float,
                        export_duplicate_pair: int,
                        select_search_target: list[str],
                        search_history: dict,
                        page_size: float,
                        progress = gr.Progress(track_tqdm=True)):

        global model

        if len(select_search_target) == 0:
            raise constants_util.GR_ERR_NO_VECTOR_DATABASE_SELECTED
        if len(select_search_target) > 1:
            raise constants_util.GR_ERR_NO_MULTIPLE_VECTOR_DATABASE_SUPPORTED
        cluster_target: str = select_search_target[0]

        vector_mgr = get_vector_db_mgr()

        vecdb = vector_mgr.get_variant(select_search_target[0])
        index: faiss.IndexFlatL2 = vecdb.db.index

        search_name = f"#{{n}} 对 {{target}} 去重，类别 <{{label}}> 数量 <{{count}}>"

        page_size=max(1, int(page_size))
        chunk_size=max(1, int(chunk_size))

        clip_model = get_clip_model()

        if clip_model.clip_model_id != "openai/clip-vit-large-patch14":
            raise gr.Error("不支持当前 CLIP 模型")

        if model is None:
            model = MLP(clip_model.model.text_embed_dim)  # CLIP embedding dim is 768 for CLIP ViT L 14
            s = torch.load("models/improved-aesthetic-predictor/ava+logos-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
            model.load_state_dict(s)
            model.to(clip_model.device)
            model.eval()

        reconstruct_embeddings = index.reconstruct_n(0, index.ntotal)
        image_embeds = torch.tensor(reconstruct_embeddings, device=clip_model.device)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        del reconstruct_embeddings

        scores = model(image_embeds)

        assert scores.shape == (index.ntotal, 1)

        t0 = time.time()

        sorted_score, indices = torch.sort(scores, dim=0, descending=True)

        indices = indices.reshape((index.ntotal,))

        #reconstruct_embeddings = index.reconstruct_batch(indices)

        #score_indices  = torch.argsort(scores, dim=0)

        dataset = Dataset.from_dict({"score": sorted_score, "indices": indices })

        dataloader = DataLoader(dataset, batch_size=chunk_size)

        non_duplicate_image_and_label = []
        duplicate_image_and_label = []

        for batch in tqdm(dataloader, desc="分块聚类去重"):
            export_label = dict()
            batch_image_embeds = torch.tensor(index.reconstruct_batch(batch["indices"]))
            batch_image_embeds = batch_image_embeds / batch_image_embeds.norm(p=2, dim=-1, keepdim=True)

            db = DBSCAN(eps=threshold, min_samples=1).fit(batch_image_embeds)

            for i, label in enumerate(db.labels_):
                label = int(label)
                image_index = int(batch["indices"][i])
                score = float(batch["score"][0][i])
                if label in export_label:
                    if export_duplicate_pair == 0 and isinstance(export_label[label], tuple):
                        export_image = export_label[label]
                        duplicate_image_and_label.append((export_image[0], f"保留图,{export_image[1]}"))
                        export_label[label] = 0
                        del export_image
                    # 记录重复的图片
                    duplicate_image_and_label.append((image_index, f"重复图,{score}"))
                else:
                    # 记录非重复图片1
                    non_duplicate_image_and_label.append((image_index, score))
                    export_label[label] = (image_index, score)

            del export_label, db, batch_image_embeds

        pages = [non_duplicate_image_and_label, duplicate_image_and_label]
        new_searchs = []

        first_search = True

        for i, label in tqdm(enumerate(pages), desc="保存分页"):
            image_and_score_list = pages[i]
            if len(image_and_score_list) == 0:
                continue

            image_and_label = []
            search_id = str(uuid.uuid4())
            search_state.search_count += 1

            for image_index, label in image_and_score_list:
                doc_uuid = vecdb.db.index_to_docstore_id[image_index]
                doc = vecdb.db.docstore.search(doc_uuid)
                filename = doc.page_content 
                if doc.metadata:
                    image_root = doc.metadata["root"]
                    filename_witout_ext = os.path.join(image_root, filename)
                else:
                    filename_witout_ext = filename
                image_and_label.append((filename_witout_ext, label))
            
            if i == 0:
                label_name = "筛选去重后"
            elif i == 1:
                if export_duplicate_pair == 0:
                    label_name = "排除的重复图片以及保留图"
                elif export_duplicate_pair == 1:
                    label_name = "排除的重复图片"
            else:
                label_name = "-"

            cur_search_name = search_name.format(n=search_state.search_count, target=cluster_target, label=label_name, count=len(image_and_label))

            if first_search:
                first_search = False
                preview_image_with_label, page_count = search_state.save_pages(search_id, image_and_label, page_size=page_size, indices=[i[0] for i in image_and_score_list], db=vecdb)
                preview_page_state = { "search_id": search_id, "page_index": 1, "page_count": page_count }
                preview_search_name = cur_search_name
            else:
                # 仅保存
                search_state.save_pages(search_id, image_and_label, page_size=page_size, indices=[i[0] for i in image_and_score_list], db=vecdb)

            del image_and_label, image_and_score_list

            new_searchs.append([cur_search_name, search_id])

        if search_history is None:
            search_history = { "search": [] }

        # 更新搜索结果列表
        search_history["search"] = new_searchs + search_history["search"] 

        del pages

        return search_state.update_viewer(
            page_state=preview_page_state,
            image_and_label=preview_image_with_label,
            search_target=None,
            search_history=search_history,
            select_search_name=preview_search_name,
            msg="分块聚类去重完毕",
            progress=progress,
        )

    chunk_size, threshold, export_duplicate_pair, btn = compolents
    
    btn.click(fn=on_deduplication, inputs=[
        chunk_size,
        threshold,
        export_duplicate_pair,
        search_state.select_search_target,
        search_state.search_history,
        search_state.page_size,
        ], outputs=search_state.get_image_viewer_outputs())