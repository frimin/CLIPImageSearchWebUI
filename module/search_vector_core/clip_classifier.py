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

DEFAULT_CLASSIFITER = """
1girl:0.7
multiple girls:0.7
old woman:0.7
1boy:0.7
multiple boys:0.7
old man:0.7
building:0.7
tree:0.7
animals:0.7
car:0.7
artistic:0.7
book:0.7
text:0.7
color block:0.7
"""

def on_gui():
    with gr.Tab(label="零样本分类"):
        gr.Markdown("通过零样本图片分类 (Zero-Shot Image Classification) 计算出每张图片对于指定提示词句的概率，并归类为最大概率所在提示词的组后，仅输出高于指定概率值的内容。\n\n"
                    "(该概率值可选默认为0) 单张图片测试可以在[这里](https://huggingface.co/openai/clip-vit-large-patch14)测试。每一个提示词句将输出一个独立的查询结果到历史记录中。")
        classifier_rule = gr.Code(label="分类标签和输出概率值(一行一个)", value=DEFAULT_CLASSIFITER, interactive=True)
        btn = gr.Button(value="分类", variant="primary")

    return [classifier_rule, btn]

def on_bind(search_state: SearchVectorPageState, compolents: list[gr.components.Component]):
    classifier_rule, btn = compolents
    
    def on_zero_shot_classification(classifier_rule: str, 
                                    select_search_target: list[str],
                                    search_history: dict,
                                    page_size: float,
                                    progress = gr.Progress(track_tqdm=True)):

        if len(select_search_target) == 0:
            raise constants_util.GR_ERR_NO_VECTOR_DATABASE_SELECTED
        if len(select_search_target) > 1:
            raise constants_util.GR_ERR_NO_MULTIPLE_VECTOR_DATABASE_SUPPORTED
        cluster_target: str = select_search_target[0]

        vector_mgr = get_vector_db_mgr()

        vecdb = vector_mgr.get_variant(select_search_target[0])
        index: faiss.IndexFlatL2 = vecdb.db.index

        page_size=max(1, int(page_size))

        classifier_rule = [i for i in classifier_rule.split("\n") if not i.isspace()]

        if len(classifier_rule) > 1000:
            raise gr.Error("输入的分类提示超过限制")

        for i, v in enumerate(classifier_rule):
            v = v.split(":")

            if len(v) == 1:
                classifier_rule[i] = (v[0], 0.0)
            else:
                classifier_rule[i] = (v[0], float(v[1]))

        for lable, p in classifier_rule:
            if p < 0:
                raise gr.Error(f"标签 {lable} 的输出概率必须大于等于 0")

        reconstruct_embeddings = index.reconstruct_n(0, index.ntotal)

        clip_model = get_clip_model()

        prompts = [i[0] for i in classifier_rule]

        clip_inputs = clip_model.processor(text=prompts, return_tensors="pt", padding=True)
        clip_inputs = clip_inputs.to(clip_model.device)

        text_embeds = clip_model.model.get_text_features(**clip_inputs) 
        text_embeds /= text_embeds.norm(p=2, dim=-1, keepdim=True) 

        image_embeds = torch.tensor(reconstruct_embeddings, device=clip_model.device)
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        del reconstruct_embeddings

        t0 = time.time()

        # cosine similarity as logits
        logit_scale = clip_model.model.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        del text_embeds, image_embeds
        logits_per_image = logits_per_text.t()
        probs = logits_per_image.softmax(dim=1)
        probs_max_index = torch.argmax(probs, dim=1)

        t1 = time.time()

        print(f"zero-shot 分类耗时: {t1-t0}")

        search_name = f"#{{n}} 对 {{target}} 分类，标签 <{{label}}> 数量 <{{count}}>"

        new_searchs = []

        label_with_image = {}

        for i in range(0, len(classifier_rule)):
            label_with_image[i] = []

        for image_index, label_index in enumerate(probs_max_index):
            label_index = int(label_index)
            p_request = classifier_rule[label_index][1]
            p = probs[image_index][label_index]
            if p >= p_request:
                label_with_image[label_index].append((image_index, float(p)))

        first_search = True

        for i, v in enumerate(tqdm(classifier_rule, desc="保存分页")):
            label, _ = v
            image_and_score_list = label_with_image[i]
            image_and_score_list = sorted(image_and_score_list, key=lambda x: x[1], reverse=True)

            if len(image_and_score_list) == 0:
                continue

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
                preview_image_with_label, page_count = search_state.save_pages(search_id, image_and_label, page_size=page_size, 
                                                                               indices=[i[0] for i in image_and_score_list], 
                                                                               db=vecdb)
                preview_page_state = { "search_id": search_id, "page_index": 1, "page_count": page_count }
                preview_search_name = cur_search_name
            else:
                # 仅保存
                search_state.save_pages(search_id, image_and_label, page_size=page_size,
                                        indices=[i[0] for i in image_and_score_list],
                                        db=vecdb)

            del image_and_label, image_and_score_list

            new_searchs.append([cur_search_name, search_id])

        if search_history is None:
            search_history = { "search": [] }

        # 更新搜索结果列表
        search_history["search"] = new_searchs + search_history["search"] 
            
        # compute image-text similarity scores
        #inputs = clip_processor(text=texts, images=image, return_tensors="pt", padding=True)

        #clip_model(
        #    input_ids=text_features["input_ids"],
        #    attention_mask=text_features["attention_mask"],
        #    pixel_values=image_features,
        #    )

        del label_with_image, probs, probs_max_index

        return search_state.update_viewer(
            page_state=preview_page_state,
            image_and_label=preview_image_with_label,
            search_target=None,
            search_history=search_history,
            select_search_name=preview_search_name,
            msg="零样本分类完毕",
            progress=progress,
        )
    
    btn.click(fn=on_zero_shot_classification, inputs=[
        classifier_rule,
        search_state.select_search_target,
        search_state.search_history,
        search_state.page_size,
        ], outputs=search_state.get_image_viewer_outputs())