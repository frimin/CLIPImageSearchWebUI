import os
from omegaconf import OmegaConf
from typing import Callable

class WebUIConfigs():
    __data = None
    def __init__(self, userdata_dir: str) -> None:
        self._callbacks = []
        self._template_config_filename= os.path.join(os.getcwd(), "data", "webui_configs.yaml") 
        self._user_config_filename = os.path.join(userdata_dir, "webui_configs.yaml") 

        if not os.path.exists(self._template_config_filename):
            raise ValueError(f"not found template options: {self._user_config_filename}")

        self._template_cfg = OmegaConf.load(self._template_config_filename)

        if not os.path.exists(self._user_config_filename):
            with open(self._user_config_filename, "w") as f:
                OmegaConf.save(self._template_cfg, f)

        user_cfg = OmegaConf.load(self._user_config_filename)
        
        self._cfg = OmegaConf.merge(self._template_cfg, user_cfg)
        self._changes_cfg = None

    def add_on_configs_changed(self, callback: Callable):
        self._callbacks.append(callback)

    def get_cfg(self):
        return self._cfg

    def get_changes_cfg(self):
        if self._changes_cfg is None:
            self._changes_cfg = OmegaConf.merge({}, self._cfg)
        return self._changes_cfg

    def apply_cfg_changes(self):
        if self._changes_cfg is None:
            return
        self._cfg = self._changes_cfg
        self._changes_cfg = None

        with open(self._user_config_filename, "w") as f:
            OmegaConf.save(self._cfg, f)

        for cb in self._callbacks:
            cb(self)

    def cancel_cfg_changes(self):
        if self._changes_cfg is None:
            return
        self._changes_cfg = None

def init_webui_configs(userdata_dir: str):
    if not os.path.exists(userdata_dir):
        os.makedirs(userdata_dir)

    if not hasattr(WebUIConfigs, "__data") or WebUIConfigs.__data is None:
        WebUIConfigs.__data = WebUIConfigs(userdata_dir)

def get_webui_configs() -> WebUIConfigs:
    return WebUIConfigs.__data