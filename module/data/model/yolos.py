import torch
from tqdm import tqdm
from module.data.webui_configs import WebUIConfigs
from transformers import YolosImageProcessor, YolosForObjectDetection

def load_yolos_model(model_cfg, create_processor=True, create_model=True, **kwargs):
    """需要一个相对原始的接口暴露给外部脚本使用"""
    if create_processor:
        processor = model_cfg.yolos_processor(model_cfg.model_id, **kwargs)
    else:
        processor = None

    if create_model:
        model = model_cfg.yolos_model(model_cfg.model_id, **kwargs)
    else:
        model = None

    return processor, model

class YOLOSModelWarpperContextModel:
    def __init__(self, processor, model) -> None:
        self.processor : YolosImageProcessor = processor 
        self.model : YolosForObjectDetection = model

class YOLOSModelWarpperContext:
    def __init__(self, wapper) -> None:
        self._wapper = wapper

    def __enter__(self) -> YOLOSModelWarpperContextModel:
        if self._wapper._model is None:
            self._wapper._load_model()
        return YOLOSModelWarpperContextModel(self._wapper._processor, self._wapper._model)
 
    def __exit__(self, *args):
        pass

class YOLOSModelWarpper:
    __data = None
    def __init__(self, webui_configs: WebUIConfigs, yolos_model_list) -> None:
        self._processor, self._model = None, None
        self.model_id: str = None
        self.device: torch.device = None
        self._yolos_model_list = yolos_model_list
        self._webui_configs: WebUIConfigs = webui_configs
        self.set_model_id(webui_configs.get_cfg().model.yolos.id)
        self._model_ctx: YOLOSModelWarpperContext = YOLOSModelWarpperContext(self)

    def set_model_id(self, model_id):
        self.release_model()
        self.model_id = model_id
        self._cfg = self._yolos_model_list.get_model(model_id)
        self.device = torch.device(self._webui_configs.get_cfg().model.yolos.device)

    def _load_model(self):
        assert self._model is None
        is_load = False
        try:
            kwargs = {}
            if self._webui_configs.get_cfg().model.offline_load:
                kwargs["local_files_only"]=True
                kwargs["force_download"] = False
            for _ in tqdm([0], desc=f"加载 {self.model_id}"):
                self._processor, self._model = load_yolos_model(self._cfg, **kwargs)
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

    def get_model(self) -> YOLOSModelWarpperContext:
        return self._model_ctx

def on_config_change(webui_configs):
    yolos = get_yolos_model()
    new_model_id = webui_configs.get_cfg().model.yolos.id
    if yolos.model_id != new_model_id:
        print(f"YOLOS [{yolos.model_id}] -> [{new_model_id}]")
        yolos.set_model_id(new_model_id)

def init_yolos_model(webui_configs: WebUIConfigs, yolos_model_list):
    if not hasattr(YOLOSModelWarpper, "__data") or YOLOSModelWarpper.__data is None:
        YOLOSModelWarpper.__data = YOLOSModelWarpper(webui_configs, yolos_model_list)
        webui_configs.add_on_configs_changed(on_config_change)

def get_yolos_model() -> YOLOSModelWarpper:
    return YOLOSModelWarpper.__data