from PIL import Image
import os
import gradio as gr
import module.utils.constants_util as constants_util
from module.data import get_clip_model_list, get_cache_root
import json
import shutil
from tqdm import tqdm
from module.utils.constants_util import IMAGE_EXTENSIONS
from typing import Callable, ParamSpec
import re
import uuid

def get_save_img_ext(format: int, ext: str):
    if format == 0:
        return ext
    elif format == 1:
        return "jpg"
    elif format == 2:
        return "png"
    else:
        raise Exception("unknown save format")

_SKIP = "skip"
_SAVE_IMAGE = "save_image"
_NO_CHANGE = "no_change"

def convert_image(source_image_filename: str, destination_image_filename: str, max_pixel: int, format: int, quality: int, check_image_cb: Callable | None) -> bool:
    with Image.open(source_image_filename) as image:
        has_change = False
        if check_image_cb is not None:
            if not check_image_cb(image):
                # 跳过处理此图像
                return _SKIP
        width, height = image.size

        if max_pixel > 0:
            if width > max_pixel and height > max_pixel:
                if width > height:
                    width_pct = max_pixel / width
                    width = max_pixel
                    height = height * width_pct
                else:
                    height_pct = max_pixel / height
                    width = width * height_pct
                    height = max_pixel

                image = image.resize((int(width), int(height)))
                has_change = True

        params = {}

        if format == 1: # JPEG
            params["format"] = "JPEG"
            image = image.convert('RGB')
            params["quality"] = quality
            has_change = True
        elif format == 2: # png
            params["format"] = "PNG"
            has_change = False
        else:
            pass

        if has_change:
            with open(destination_image_filename, "w") as f:
                image.save(f, optimize=True, **params)
                return _SAVE_IMAGE
        return _NO_CHANGE

def save_query_image_to_dir(
    page_state: dict(),
    search_history: dict(),
    copy_type: int,
    outdir: str,
    start_page_index: float,
    end_page_index: float,
    max_export_count: float,
    copy_same_name_ext: str,
    random_new_name: bool,
    skip_img_filesize: float,
    skip_img_pixel: float,
    skip_img_scale: float,
    max_pixel: float,
    format: int,
    quality: float,
    progress = gr.Progress(track_tqdm=True)
):
    if page_state is None:
        raise constants_util.NO_QUERY_RESULT_ERROR

    clip_model_list = get_clip_model_list()

    search_id = page_state["search_id"]

    start_page_index = int(max(start_page_index, 1))
    end_page_index = int(end_page_index)

    quality = min(max(int(quality), 0), 100)

    skip_img_filesize = int(skip_img_filesize) * 1024

    max_export_count = max(int(max_export_count), 0)

    save_search = []

    if copy_type == 0:
        # 保存当前查询
        for search_name, id in search_history["search"]:
            if id == search_id:
                save_search.append([search_name, id])      
    elif copy_type == 1:
        # 保存所有查询
        save_search = search_history["search"]

    def check_img(image: Image.Image):
        width, height = image.size
        if width < skip_img_pixel or height < skip_img_pixel:
            return False
        if skip_img_scale > 0:
            if (float(width) / float(height)) > skip_img_scale:
                return False # 跳过
            if (float(height) / float(width)) > skip_img_scale:
                return False # 跳过
        return True # 继续执行

    copy_same_name_ext = [i.strip() for i in copy_same_name_ext.split(',') ]

    for save_search_name, save_search_id in tqdm(save_search, desc="保存查询"):
        cache_root = os.path.join(get_cache_root().cache_root, "search_id", save_search_id)

        # 获取路径安全的名字
        valid_save_search_name = re.sub('[^\w_.)( -]', '', save_search_name)

        search_outdir = os.path.join(outdir, valid_save_search_name)
        export_count = 0

        with open(os.path.join(cache_root, "pages_meta.json"), "r") as f:
            page_state = json.load(f)

        with open(os.path.join(cache_root, "pages_index.json"), "r") as f:
            page_info = json.load(f)

        if end_page_index < 0:
            search_end_page_index = page_state["page_count"]

        search_start_page_index = min(start_page_index, search_end_page_index)

        if not os.path.exists(search_outdir):
            os.makedirs(search_outdir)

        with open(os.path.join(cache_root, "pages.json"), "r") as f:
            for cur_page_index in tqdm(list(range(search_start_page_index, search_end_page_index + 1)), desc=f"保存分页: {save_search_name}"):
                if cur_page_index < search_start_page_index:
                    continue
                if cur_page_index > search_end_page_index:
                    continue
                if not (max_export_count != 0 and export_count >= max_export_count):
                    page_pos_start, page_pos_end = page_info[cur_page_index - 1]
                    cur_fd_pos = f.tell()
                    if cur_fd_pos != page_pos_start:
                        f.seek(page_pos_start)
                    content = f.read(page_pos_end - page_pos_start)
                    files = json.loads(content)

                    for filename, lable in files:
                        if max_export_count != 0 and export_count >= max_export_count:
                            break
                        find_img = False
                        for image_ext in IMAGE_EXTENSIONS:
                            base_name = os.path.basename(filename)
                            if random_new_name:
                                out_base_name = f"[{cur_page_index},{lable}] {uuid.uuid4()}"
                            else:
                                out_base_name = f"[{cur_page_index},{lable}] {base_name}"
                            filename_with_ext = filename + image_ext
                            if os.path.exists(filename_with_ext):
                                find_img = True
                                save_image_ext = get_save_img_ext(format, image_ext)
                                save_to_filename = os.path.join(search_outdir, f"{out_base_name}{save_image_ext}")
                                file_stats = os.stat(filename_with_ext)
                                if file_stats.st_size < skip_img_filesize:
                                    break
                                else:
                                    save_state = convert_image(
                                        source_image_filename=filename_with_ext,
                                        destination_image_filename=save_to_filename,
                                        max_pixel=max_pixel,
                                        format=format,
                                        quality=quality,
                                        check_image_cb=check_img)
                                    if save_state is _SKIP:
                                        break
                                    export_count += 1
                                    if save_state is _NO_CHANGE:
                                        shutil.copyfile(filename_with_ext, save_to_filename)
                                # 拷贝同名的其它扩展名的文件
                                for additional_ext in copy_same_name_ext:
                                    additional_filename_with_ext = filename + additional_ext
                                    if os.path.exists(additional_filename_with_ext):
                                        additional_save_to_filename = os.path.join(search_outdir, f"{out_base_name}{additional_ext}")
                                        shutil.copyfile(additional_filename_with_ext, additional_save_to_filename)
                                    
                                # 拷贝 embed 文件
                                for short_name in clip_model_list.get_short_names():
                                    embed_ext = f".{short_name}.embed.pt"
                                    embed_filename_with_ext = filename + embed_ext
                                    if os.path.exists(additional_filename_with_ext):
                                        additional_save_to_filename = os.path.join(search_outdir, f"{out_base_name}{embed_ext}")
                                        shutil.copyfile(embed_filename_with_ext, additional_save_to_filename)

                                break


                        if not find_img:
                            print(f"image file missing: {filename}")
                    del files, content

    return [
        "完成", 
        outdir,
    ]

def save_select_image(outdir, 
                      image_file_with_lable_list: list[tuple[str, str]], 
                      select_index: float,
                      copy_same_name_ext: str,):
    if image_file_with_lable_list is None:
        raise constants_util.NO_QUERY_RESULT_ERROR
    select_index = int(select_index)
    if select_index < 0:
        raise gr.Error("未选择图片")
    image_filename, _ = image_file_with_lable_list[select_index]

    image_basename = os.path.basename(image_filename)
    filename, _ = os.path.splitext(image_basename)
    out_filename = os.path.join(outdir, image_basename)

    shutil.copyfile(image_filename, os.path.join(outdir, image_basename))

    for additional_ext in copy_same_name_ext:
        additional_filename_with_ext = filename + additional_ext
        additional_save_to_filename = os.path.join(outdir, f"{image_basename}{additional_ext}")
        if os.path.exists(additional_filename_with_ext):
            shutil.copyfile(additional_filename_with_ext, additional_save_to_filename)

    return f"已保存到 {out_filename}"