import torch
from tqdm import tqdm
from module.data.webui_configs import WebUIConfigs
from transformers import (
    CLIPProcessor, 
    CLIPModel, 
)

def load_clip_model(clip_model_cfg, create_processor=True, create_model=True, **kwargs):
    """需要一个相对原始的接口暴露给外部脚本使用"""
    if create_processor:
        processor = clip_model_cfg.clip_processor(clip_model_cfg.model_id, **kwargs)
    else:
        processor = None

    if create_model:
        model = clip_model_cfg.clip_model(clip_model_cfg.model_id, **kwargs)
    else:
        model = None

    return processor, model

class CLIPWarpperContextModel:
    def __init__(self, processor, model) -> None:
        self.processor : CLIPProcessor = processor 
        self.model : CLIPModel = model

class CLIPWarpperContext:
    def __init__(self, clip_wapper) -> None:
        self._clip_wapper = clip_wapper

    def __enter__(self) -> CLIPWarpperContextModel:
        if self._clip_wapper._model is None:
            self._clip_wapper._load_model()
        return CLIPWarpperContextModel(self._clip_wapper._processor, self._clip_wapper._model)
 
    def __exit__(self, *args):
        pass

class CLIPWarpper:
    __data = None
    def __init__(self, webui_configs: WebUIConfigs, clip_model_list) -> None:
        self._processor, self._model = None, None
        self.clip_model_id: str = None
        self.device: torch.device = None
        self._clip_model_list = clip_model_list
        self._webui_configs: WebUIConfigs = webui_configs
        self.set_clip_model_id(webui_configs.get_cfg().clip_model_id)
        self._model_ctx: CLIPWarpperContext = CLIPWarpperContext(self)

    def set_clip_model_id(self, clip_model_id):
        self.release_model()
        self.clip_model_id = clip_model_id
        self._cfg = self._clip_model_list.get_model(clip_model_id)
        self.device = torch.device(self._webui_configs.get_cfg().model.clip.device)

    def get_project_url(self) -> str:
        return self._cfg.project_url

    def get_model_subpath(self):
        return self.clip_model_id.replace("/", "---")

    def get_dim(self):
        return self._cfg.dim

    def get_short_name(self):
        return self._cfg.short_name

    def _load_model(self):
        assert self._model is None
        is_load = False
        try:
            kwargs = {}
            if self._webui_configs.get_cfg().model.offline_load:
                kwargs["local_files_only"]=True
                kwargs["force_download"] = False
            for _ in tqdm([0], desc=f"加载 {self.clip_model_id}"):
                self._processor, self._model = load_clip_model(self._cfg, **kwargs)
                #clip_model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16)
                self._model.requires_grad_(False)
                self._model = self._model.eval()
                self._model = self._model.to(self.device)
                is_load = True
        finally:
            if not is_load:
                self.release_model()

    def is_load_model(self) -> bool:
        return self._model is not None

    def release_model(self):
        del self._processor, self._model
        self._processor, self._model = None, None

    def get_model(self) -> CLIPWarpperContext:
        return self._model_ctx
    

def on_config_change(webui_configs):
    clip = get_clip_model()
    new_clip_model_id = webui_configs.get_cfg().clip_model_id
    if clip.clip_model_id != new_clip_model_id:
        print(f"CLIP [{clip.clip_model_id}] -> [{new_clip_model_id}]")
        clip.set_clip_model_id(new_clip_model_id)

def init_clip_model(webui_configs: WebUIConfigs, clip_model_list):
    if not hasattr(CLIPWarpper, "__data") or CLIPWarpper.__data is None:
        CLIPWarpper.__data = CLIPWarpper(webui_configs, clip_model_list)
        webui_configs.add_on_configs_changed(on_config_change)

def get_clip_model() -> CLIPWarpper:
    return CLIPWarpper.__data