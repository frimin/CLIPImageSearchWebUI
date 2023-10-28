import os
import hydra
from omegaconf import OmegaConf, ListConfig

class ModelListBase():
    def __init__(self, cfg) -> None:
        self._cfg = cfg

    def get_model(self, model_id):
        for i in self._cfg:
            if i.model_id == model_id:
                return i
        raise ValueError(f"not find model id: {model_id}")

    def get_cfg(self):
        return self._cfg

class CLIPModelList(ModelListBase):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._short_names = []

        for i in self._cfg:
            self._short_names.append(i.short_name)

    def get_short_names(self):
        return self._short_names

class YOLOSModelList(ModelListBase):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

class YOLOModelList(ModelListBase):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)

class ModelCenter():
    __data = None
    def __init__(self) -> None:
        self._config_filename = os.path.join(os.getcwd(), "data", "models.yaml") 
        self._cfg = OmegaConf.load(self._config_filename)
        self._cfg = hydra.utils.instantiate(self._cfg)
        assert isinstance(self._cfg.clip, ListConfig)

        self.clip = CLIPModelList(self._cfg.clip)
        self.yolos = YOLOSModelList(self._cfg.yolos)
        self.yolo = YOLOModelList(self._cfg.yolo)

def init_model_list():
    if not hasattr(ModelCenter, "__data") or ModelCenter.__data is None:
        ModelCenter.__data = ModelCenter()

def get_clip_model_list() -> CLIPModelList:
    return ModelCenter.__data.clip

def get_yolos_model_list() -> YOLOSModelList:
    return ModelCenter.__data.yolos

def get_yolo_model_list() -> YOLOModelList:
    return ModelCenter.__data.yolo