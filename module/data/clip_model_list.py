import os
import hydra
from omegaconf import OmegaConf, ListConfig

class CLIPModelList():
    __data = None
    def __init__(self) -> None:
        self._config_filename = os.path.join(os.getcwd(), "data", "clip_models.yaml") 
        self._cfg = OmegaConf.load(self._config_filename)
        assert isinstance(self._cfg, ListConfig)
        self._cfg = hydra.utils.instantiate(self._cfg)
        self._short_names = []

        for i in self._cfg:
            self._short_names.append(i.short_name)

    def get_cfg(self):
        return self._cfg

    def get_model_cfg_by_model_id(self, model_id):
        for i in self._cfg:
            if i.model_id == model_id:
                return i
        raise ValueError(f"not find clip model id: {model_id}")

    def get_short_names(self):
        return self._short_names

def init_clip_model_list():
    if not hasattr(CLIPModelList, "__data") or CLIPModelList.__data is None:
        CLIPModelList.__data = CLIPModelList()

def get_clip_model_list() -> CLIPModelList:
    return CLIPModelList.__data