import os
import gradio as gr
from module.search_vector_core.search_state import SearchVectorPageState
from module.data import get_vector_db_mgr
import faiss
import numpy as np
import time
import uuid
from sklearn.cluster import KMeans, DBSCAN
import module.utils.constants_util as constants_util
from tqdm import tqdm
import torch

CLUSTER_METHOD = ["DBSCAN", "K-Means", ]

def on_gui():
    with gr.Tab(label="聚类"):
        with gr.Row():
            gr.Markdown("提供 DBSCAN 和 K-Means 的聚类方法，对图片嵌入向量进行聚类 (embedding clustering)。\n\n"
                        "聚类结果的的每个簇群将分别添加到查询记录中，聚类算法区别详细见[此处](https://scikit-learn.org/stable/modules/clustering.html)。")
        with gr.Row():
            cluster_method = gr.Dropdown(label="聚类方法", info="选择需要的聚类方法", choices=CLUSTER_METHOD, value=CLUSTER_METHOD[0], interactive=True)
            n_clusters = gr.Number(label="簇群数量", info="K-Means 聚类方法中期望的输出簇群数量 [2,64]", value=8, minimum=2, maximum=64)
            threshold = gr.Slider(label="DBSCAN 阈值", info="DBSCAN 聚类方法中的阈值，较小的阈值让更相似的图片分为一组。[0.05,1]", value=0.45, minimum=0.05, maximum=1, step=0.05)
            top_n = gr.Number(label="Top N", info="分组完毕后将会按照类的数量由高到低排序,指定此值返回最前N组类。[1,10000]", value=20, minimum=1, maximum=10000)
            min_count = gr.Number(label="输出最小簇群", info="簇群内实体数量小于此值时不再输出结果", value=2, minimum=1)
        gr.Markdown("提供图片或文本提示，将聚类后的簇群的均值按照提示进行相似度排序，而非簇群实体数量排序。", visible=False)
        with gr.Row(visible=False):
            search_image = gr.Image(label="图片提示", type="pil", interactive=True)
            search_text = gr.TextArea(label="文本提示", value="person, happy", info="查询提示文本，仅限于英文", interactive=True)
        btn = gr.Button(value="聚类", variant="primary")

    return [btn, cluster_method, n_clusters, threshold, top_n, min_count, search_image, search_text]

def on_bind(search_state: SearchVectorPageState, compolents: list[gr.components.Component]):
    def on_repeat_query(select_search_target: list[str], 
                        search_history: dict, 
                        cluster_method: str,
                        n_clusters: float,
                        threshold: float, 
                        top_n: float,
                        min_count: float,
                        page_size: float,
                        progress = gr.Progress(track_tqdm=True)):
        if len(select_search_target) == 0:
            raise constants_util.GR_ERR_NO_VECTOR_DATABASE_SELECTED
        if len(select_search_target) > 1:
            raise constants_util.GR_ERR_NO_MULTIPLE_VECTOR_DATABASE_SUPPORTED
        cluster_target: str = select_search_target[0]

        search_name = f"#{{n}} 对 {{target}} 聚类，类别 <{{label}}> 数量 <{{count}}>"

        vector_mgr = get_vector_db_mgr()

        vecdb = vector_mgr.get_variant(select_search_target[0])
        index: faiss.IndexFlatL2 = vecdb.db.index

        #reconstruct_embeddings = index.reconstruct_batch(np.arange(0, 10000))
        reconstruct_embeddings = index.reconstruct_n(0, index.ntotal)
        reconstruct_embeddings = torch.tensor(reconstruct_embeddings)
        reconstruct_embeddings = reconstruct_embeddings / reconstruct_embeddings.norm(p=2, dim=-1, keepdim=True)

        #scores, indices = db.db.index.search(np.array(reconstruct_embeddings, dtype=np.float32), 4)
        threshold=max(threshold, 0.01)
        page_size=max(1, int(page_size))
        min_count=max(1, int(min_count))
        top_n=max(0,int(top_n))
        n_clusters=max(1,int(n_clusters))

        if index.ntotal > 100000:
            raise gr.Error(f"实体数量过多，当前数量: {index.ntotal} 最大支持: {100000}")

        if cluster_method == "K-Means":
            cluster_method_result = KMeans(n_clusters=n_clusters).fit(reconstruct_embeddings)
        elif cluster_method == "DBSCAN":
            cluster_method_result = DBSCAN(eps=threshold, min_samples=1).fit(reconstruct_embeddings)
        del reconstruct_embeddings
        labels = cluster_method_result.labels_

        labels_counter: dict[int, int] = {}
        labels_indices : dict[int, list[int]] = {}

        for i, label in enumerate(labels):
            label = int(label)
            if label == -1:
                continue
            if label in labels_counter:
                labels_counter[label]+=1
            else:
                labels_counter[label]=1
            
            if label in labels_indices :
                labels_indices[label].append(i)
            else:
                labels_indices[label] = [i]

        del cluster_method_result

        if len(labels_indices) == 0:
            raise gr.Error("无法聚类，输入数据太过于相似，尝试调整更低的阈值再试")

        sorted_labels_counter = sorted(labels_counter.items(), key=lambda x: x[1], reverse=True)
        sorted_labels_counter = [i for i in sorted_labels_counter if i[1] >= min_count]

        if len(sorted_labels_counter) == 0:
            raise gr.Error("聚类后无输出结果，尝试调整更大的阈值，且调小'最小组'的值")

        if top_n == 0:
            top_labels = sorted_labels_counter
        else:
            top_labels = sorted_labels_counter[:top_n]

        new_searchs = []

        index_label = 0

        for i, v in enumerate(tqdm(top_labels, desc="保存分页")):
            (label, count) = v
            indices = labels_indices[label]

            image_and_label = []
            search_id = str(uuid.uuid4())
            search_state.search_count += 1

            for j in indices:
                doc_uuid = vecdb.db.index_to_docstore_id[j]
                doc = vecdb.db.docstore.search(doc_uuid)
                filename = doc.page_content 
                if doc.metadata:
                    image_root = doc.metadata["root"]
                    filename_witout_ext = os.path.join(image_root, filename)
                else:
                    filename_witout_ext = filename
                index_label+=1
                image_and_label.append((filename_witout_ext, f"{index_label}"))

            cur_search_name = search_name.format(n=search_state.search_count, target=cluster_target, label=label, count=count)

            if i == 0:
                preview_image_with_label, page_count = search_state.save_pages(search_id, image_and_label, page_size=page_size, indices=indices, db=vecdb)
                preview_page_state = { "search_id": search_id, "page_index": 1, "page_count": page_count }
                preview_search_name = cur_search_name
            else:
                # 仅保存
                search_state.save_pages(search_id, image_and_label, page_size=page_size, indices=indices, db=vecdb)

            del image_and_label

            new_searchs.append([cur_search_name, search_id])

        if search_history is None:
            search_history = { "search": [] }

        # 更新搜索结果列表
        search_history["search"] = new_searchs + search_history["search"] 
        # Number of clusters in labels, ignoring noise if present.
        #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        #n_noise_ = list(labels).count(-1)

        del sorted_labels_counter, top_labels, new_searchs, labels

        #print("Estimated number of clusters: %d" % n_clusters_)

        return search_state.update_viewer(
            page_state=preview_page_state,
            image_and_label=preview_image_with_label,
            search_target=None,
            search_history=search_history,
            select_search_name=preview_search_name,
            msg="聚类完毕",
            progress=progress,
        )
        

    btn, cluster_method, n_clusters, threshold, top_n, min_count, search_image, search_text = compolents

    btn.click(fn=on_repeat_query, inputs=[
        search_state.select_search_target,
        search_state.search_history,
        cluster_method,
        n_clusters,
        threshold,
        top_n,
        min_count,
        search_state.page_size,
    ], outputs=search_state.get_image_viewer_outputs())