import os
import argparse
from multiprocessing import Pool
from tqdm import tqdm
from module.core.src_datasets import SrcDataset
from omegaconf import OmegaConf
from module.data import init_model_list, get_yolo_model_list 
from module.foundation.image_augmentation import (
    ImageAugmentationPipeline, 
    ImageAugmentationActions, 
    PresetsList
)

def args_builder():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'cfg', type=str, help='yaml config file', default=None,
    )

    parser.add_argument(
        '--indir', type=str, help='Image output folder', default=None, required=True,
    )

    parser.add_argument(
        '--outdir', type=str, help='Image output folder', default=None, required=True,
    )

    parser.add_argument(
        '--device', type=str, help='all model device', default=r"cuda",
    )

    parser.add_argument(
        '--num_workers', type=int, help='Num workers for process image', default=2,
    )

    parser.add_argument(
        '--outdir_with_step', action='store_true', default=False
    )

    args = parser.parse_args()

    return args

pipe = None

def worker(file):
    global pipe
    pipe.process_image(file)

def init_pipe(args):
    global pipe
    cfg = OmegaConf.load(args.cfg)
    PresetsList.init()
    init_model_list()
    yolo_model_list = get_yolo_model_list()
    ImageAugmentationActions.init(yolo_model_list)
    pipe = ImageAugmentationPipeline(cfg, args.indir, args.outdir, args.outdir_with_step)

if __name__ != "__main__":
    print(f"Start worker: [{os.getpid()}]")
    args = args_builder()
    init_pipe(args)
else:
    args = args_builder()

    files = SrcDataset(args.indir).get_files()

    if len(files) == 0:
        raise ValueError("no image input")

    files_len = len(files)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    init_pipe(args)

    num_workers = max(1, min(64, args.num_workers))

    if files_len < num_workers:
        num_workers = files_len

    with Pool(processes=num_workers) as p:
        with tqdm(total=files_len) as pbar:
            for _ in p.imap_unordered(worker, files):
                pbar.update()


