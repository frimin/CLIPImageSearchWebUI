import os
import shutil

PREVIEW_CACHE_DIR = os.path.abspath("./app_cache")

if not os.path.exists(PREVIEW_CACHE_DIR):
    os.mkdir(PREVIEW_CACHE_DIR)

def get_cache_dir():
    return PREVIEW_CACHE_DIR

def clear_cache_dir():
    if os.path.exists(PREVIEW_CACHE_DIR):
        shutil.rmtree(PREVIEW_CACHE_DIR)
        os.mkdir(PREVIEW_CACHE_DIR)
