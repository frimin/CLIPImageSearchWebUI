import os
import numpy as np
import torch
import json
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import FakeEmbeddings
from langchain.docstore.in_memory import InMemoryDocstore
from module.core.src_datasets import SrcDataset
from module.data.model.clip import get_clip_model, CLIPWarpper
from module.data.webui_configs import WebUIConfigs
from tqdm import tqdm
import gradio as gr

class VectorDatabase():
    fake_embeddings = FakeEmbeddings(size=1)

    def __init__(self) -> None:
        self.loaded_path: set = set()
        self.db: FAISS = None

    def is_load_path(self, image_root: str):
        return image_root in self.loaded_path

    def load_path(self, clip_model: CLIPWarpper, db_path: str, image_root: str, dim: int):
        if image_root:
            meta = {"root": image_root}
        if not os.path.exists(db_path):
            self.load_from_meta(clip_model, image_root, dim)
            return

        db = FAISS.load_local(db_path, embeddings=self.fake_embeddings)

        if image_root:
            for v in db.docstore._dict.values():
                v.metadata = meta

        if self.db is None:
            self.db = db
        else:
            self.db.merge_from(db)
        if image_root:
            self.loaded_path.add(image_root)

    def load_from_meta(self, clip_model: CLIPWarpper, image_root: str, dim: int):
        meta = {"root": image_root}

        if image_root[-1] == '\\' or image_root[-1] == '/':
            image_root = image_root[:-1]

        prefix_len = len(image_root) + 1

        embed_file_tail=f".{clip_model.get_short_name()}.embed.pt"
        embed_file_tail_len = len(embed_file_tail)

        ds = SrcDataset(root=image_root, endswith_str=embed_file_tail)
        loader = torch.utils.data.DataLoader(ds, batch_size=5000)

        device = torch.device('cpu')

        for batch in tqdm(loader, desc=f"从元信息加载: {image_root}"):
            data = []
            for i, embed_filename in enumerate(batch):
                embedding = torch.load(embed_filename, map_location=device)

                if len(embedding) != dim:
                    raise gr.Error("图片的向量文件与当前选择的CLIP模型维度不匹配")

                embedding /= embedding.norm(dim=-1, keepdim=True) 
                image_name = embed_filename[:-embed_file_tail_len]
                image_name = image_name[prefix_len:]
                data.append((image_name, embedding.tolist()))

            db_new = FAISS.from_embeddings(text_embeddings=data, embedding=self.fake_embeddings)

            for v in db_new.docstore._dict.values():
                v.metadata = meta

            if self.db is None:
                self.db = db_new
            else:
                self.db.merge_from(db_new)
                del db_new

            del data

        self.loaded_path.add(image_root)

    def get_entity_count(self):
        return self.db.index.ntotal

    def get_db_count(self):
        return len(self.loaded_path)

class VectorDatabaseManager():
    __data = None
    def __init__(self) -> None:
        self.vecdb_names: list[str] = []
        self.vecdb_variants: dict[str,VectorDatabase] = {}
        self._latest_clip_model_id = None

    def load_path(self, clip_model: CLIPWarpper, db_path: str, image_root: str, variant: str = None, dim: int = None):
        db_path = db_path.strip()

        if variant is None:
            variant = "default"

        if not db_path or len(db_path) == 0:
            raise ValueError(f"无效的路径")

        if self._latest_clip_model_id is None:
            self._latest_clip_model_id = get_clip_model().clip_model_id
        else:
            if self._latest_clip_model_id != get_clip_model().clip_model_id: 
                raise gr.Error(f"当前CLIP模型为 {get_clip_model().clip_model_id}, 与库模型 {self._latest_clip_model_id} 不匹配")
        
        db: VectorDatabase = None
        is_new_db = False

        if variant in self.vecdb_variants:
            db = self.vecdb_variants[variant]
            if image_root and db.is_load_path(image_root=image_root):
                # 跳过已加载的
                return
        else:
            db = VectorDatabase()
            is_new_db = True

        db.load_path(clip_model, db_path, image_root, dim)
        
        if db.db is None:
            raise ValueError(f"load vec db error: db_path={db_path}, image_root={image_root}")

        self.vecdb_variants[variant] = db
        if is_new_db:
            self.vecdb_names.append(variant)

    def unload_all(self):
        self._latest_clip_model_id = None
        del self.vecdb_variants
        del self.vecdb_names
        self.vecdb_names = []
        self.vecdb_variants = {}

    def is_empty(self):
        return len(self.vecdb_variants) == 0

    def get_master_names(self):
        return list(self.vecdb_names)

    def get_entity_count(self):
        n = 0
        for v in self.vecdb_variants.values():
            n+=v.get_entity_count()
        return n
    
    def get_db_count(self):
        n = 0
        for v in self.vecdb_variants.values():
            n+=v.get_db_count()
        return n

    def get_variant(self, variant: str) -> VectorDatabase:
        return self.vecdb_variants[variant]

    def is_exsits_variant(self, variant: str) -> bool:
        return variant in self.vecdb_variants

    def search(self, embedding: list[float], top_k : int, create_db: bool = False, variant: str = "default") -> tuple[list[tuple[str, float]], FAISS]:
        """嵌入搜索"""

        results = [] 

        db = self.vecdb_variants[variant]

        top_k = min(top_k, db.db.index.ntotal)

        scores, indices = db.db.index.search(np.array([embedding], dtype=np.float32), top_k)
        for j, i in enumerate(indices[0]):
            # top k 超过向量库本身的长度搜索的结果则为-1，跳过
            if i == -1:
                continue
            uuid = db.db.index_to_docstore_id[i]
            doc = db.db.docstore.search(uuid)
            filename = doc.page_content 

            if doc.metadata:
                image_root = doc.metadata["root"]
                filename_witout_ext = os.path.join(image_root, filename)
            else:
                filename_witout_ext = filename

            results.append((filename_witout_ext, float(scores[0][j])))

        return results, indices[0], db

    def load_vector_db_from_lines(self, lines: list[str]):
        clip_model = get_clip_model()

        for path in tqdm(lines, desc="加载向量库"):
            variant = None
            if path.startswith("<"):
                end_pos = path.find(">")
                if end_pos >= 0:
                    variant = path[1:end_pos]
                    path = path[end_pos+1:]
            db_path = os.path.join(path, ".vec_db", clip_model.get_model_subpath())
            self.load_path(clip_model=clip_model, db_path=db_path, image_root=path, variant=variant, dim=clip_model.get_dim())


def init_vector_db(webui_configs: WebUIConfigs):
    if not hasattr(VectorDatabaseManager, "__data") or VectorDatabaseManager.__data is None:
        VectorDatabaseManager.__data = VectorDatabaseManager()
        cfg = webui_configs.get_cfg()
        if cfg.vector_db.load_start_count >= 0 and len(cfg.librarys) > 0:
            load_librarys = list(cfg.librarys)
            if cfg.vector_db.load_start_count > 0:
                load_librarys = load_librarys[:cfg.vector_db.load_start_count]
            loaded = False
            try:
                VectorDatabaseManager.__data.load_vector_db_from_lines(load_librarys)
                loaded = True
            except ValueError as e:
                print(e)
                VectorDatabaseManager.__data.unload_all()
            finally:
                if not loaded:
                    VectorDatabaseManager.__data.unload_all()

def get_vector_db_mgr() -> VectorDatabaseManager:
    return VectorDatabaseManager.__data