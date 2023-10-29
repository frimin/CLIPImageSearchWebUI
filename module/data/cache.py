import os
from uuid import uuid4
import shutil

class CacheRoot():
    __data = None
    def __init__(self, userdata_dir: str) -> None:
        self._userdata_dir = os.path.abspath(userdata_dir)
        self.cache_root = os.path.join(self._userdata_dir, "cache")
        self.temporary_root = os.path.join(self._userdata_dir, "cache", "temporary")
        self._clear_dirs()
        self._make_dirs()

    def _clear_dirs(self):
        if os.path.exists(self.cache_root):
            shutil.rmtree(self.cache_root)

    def _make_dirs(self):
        if not os.path.exists(self.cache_root):
            os.mkdir(self.cache_root)

        if not os.path.exists(self.temporary_root):
            os.mkdir(self.temporary_root)

    def create_temporary_filename(self, file_ext = None):
        if file_ext is None:
            return os.path.join(self.temporary_root, str(uuid4()))
        else:
            if file_ext[0] != ".":
                file_ext = "." + file_ext
            return os.path.join(self.temporary_root, str(uuid4()) + file_ext)

def init_cache_root(userdata_dir: str):
    if not hasattr(CacheRoot, "__data") or CacheRoot.__data is None:
        CacheRoot.__data = CacheRoot(userdata_dir)

def get_cache_root() -> CacheRoot:
    return CacheRoot.__data