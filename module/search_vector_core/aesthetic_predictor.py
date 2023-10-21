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
from .aesthetic_predictor_mlp import MLP

model = None

def on_gui():
    with gr.Tab(label="美学评分器"):
        gr.Markdown("基于CLIP嵌入作为输入训练的简单神经网络审美分数预测器 (人们平均喜欢图像的程度), 当前使用 ava+logos-l14-linearMSE 模型。\n\n"
                    "评分的每个区间分别添加到查询记录中，并且每个记录中的实体将以评分从大到小排列。")
        score_interval = gr.Number(label="", value=1, minimum=0.1, maximum=5, visible=False)
        btn = gr.Button(value="评分", variant="primary")

    return [score_interval, btn]

def on_bind(search_state: SearchVectorPageState, compolents: list[gr.components.Component]):
    def on_aesthetic_predictor(score_interval: float, 
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

        page_size=max(1, int(page_size))

        clip_model = get_clip_model()

        if clip_model.clip_model_id != "openai/clip-vit-large-patch14":
            raise gr.Error("不支持当前 CLIP 模型")

        if model is None:
            model = MLP(clip_model.get_dim()) 
            s = torch.load("models/improved-aesthetic-predictor/ava+logos-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo
            model.load_state_dict(s)
            model.to(clip_model.device)
            model.eval()

        reconstruct_embeddings = index.reconstruct_n(0, index.ntotal)
        image_embeds = torch.tensor(reconstruct_embeddings, device=clip_model.device)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        del reconstruct_embeddings

        scores = model(image_embeds)

        score_classifier = [ "最佳审美", "正常审美", "较差审美", "异常"]

        search_name = f"#{{n}} 对 {{target}} 评分，标签 <{{label}}> 数量 <{{count}}>"

        new_searchs = []

        label_with_image = {}

        for i, _ in enumerate(score_classifier):
            label_with_image[i] = []

        for image_index, score in enumerate(scores):
            score = float(score)
            if score >= 6.675:
                score_idx = 3
            elif score >= 6:
                score_idx = 0
            elif score >= 5:
                score_idx = 1
            else:
                score_idx = 2
            
            label_with_image[score_idx].append((image_index, score))

        del scores

        first_search = True

        for i, label in tqdm(enumerate(score_classifier), desc="保存分页"):
            image_and_score_list = label_with_image[i]
            if len(image_and_score_list) == 0:
                continue
            image_and_score_list = sorted(image_and_score_list, key=lambda x: x[1], reverse=True)

            image_and_label = []
            search_id = str(uuid.uuid4())
            search_state.search_count += 1

            for image_index, label_probs in image_and_score_list:
                doc_uuid = vecdb.db.index_to_docstore_id[image_index]
                doc = vecdb.db.docstore.search(doc_uuid)
                filename = doc.page_content 
                if doc.metadata:
                    image_root = doc.metadata["root"]
                    filename_witout_ext = os.path.join(image_root, filename)
                else:
                    filename_witout_ext = filename
                image_and_label.append((filename_witout_ext, label_probs))

            cur_search_name = search_name.format(n=search_state.search_count, target=cluster_target, label=label, count=len(image_and_label))

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

        del label_with_image

        return search_state.update_viewer(
            page_state=preview_page_state,
            image_and_label=preview_image_with_label,
            search_target=None,
            search_history=search_history,
            select_search_name=preview_search_name,
            msg="美学评分分类完毕",
            progress=progress,
        )

    score_interval, btn = compolents
    
    btn.click(fn=on_aesthetic_predictor, inputs=[
        score_interval,
        search_state.select_search_target,
        search_state.search_history,
        search_state.page_size,
        ], outputs=search_state.get_image_viewer_outputs())