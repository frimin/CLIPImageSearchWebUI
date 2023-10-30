from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from transformers import CLIPProcessor
from datasets import Image, Dataset
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from module.data import init_model_list, get_clip_model_list, load_clip_model
from module.core.src_datasets import SrcDataset
import os

IMAGE_EXT = (".png", ".jpg", ".jpeg",".webp")

clip_processor = None
embed_file_tail = None

def encode(batch):
    global clip_processor, embed_file_tail
    ret = clip_processor(images=batch["image"], return_tensors="pt", padding=True) 
    return { 
        "pixel_values": ret["pixel_values"], 
        "embed_file": [ os.path.splitext(i)[0] + embed_file_tail for i in batch["path"] ],
        "image_size": [(i.size) for i in batch["image"] ], 
    }

def args_builder():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--indir', type=str, action='append', help='Input image folder', default=None, required=True,
    )

    parser.add_argument(
        '--clip_model_id', type=str, help='CLIP model ID', default=r"openai/clip-vit-large-patch14",
    )

    parser.add_argument(
        '--device', type=str, help='CLIP model device', default=r"cuda",
    )

    parser.add_argument(
        '--batch_size', type=int, help='Batch size', default=1,
    )

    parser.add_argument(
        '--num_workers', type=int, help='Num workers for process image', default=0,
    )

    parser.add_argument(
        '--img_ext', type=str, nargs='+', help='Processed image extensions', default=IMAGE_EXT,  required=False,
    )

    args = parser.parse_args()

    return args

def create_clip_model(clip_model_id, create_model=True, **kwargs):
    init_model_list()
    model_list = get_clip_model_list()
    model_cfg = model_list.get_model(clip_model_id)
    processor, model = load_clip_model(model_cfg, create_model=create_model, **kwargs)
    return processor, model, model_cfg

if __name__ != "__main__":
    print(f"Start worker: [{os.getpid()}]")
    args = args_builder()
    clip_processor, _, model_cfg = create_clip_model(args.clip_model_id, create_model=False, local_files_only=True, force_download=False)
    embed_file_tail=f".{model_cfg.short_name}.embed.pt"
else:
    from transformers import CLIPModel

    args = args_builder()    

    print(f"Load model: {args.clip_model_id}")

    clip_processor, clip_model, model_cfg = create_clip_model(args.clip_model_id)
    clip_model.requires_grad_(False)
    clip_model = clip_model.eval().to(args.device)

    image_files = []

    embed_file_tail=f".{model_cfg.short_name}.embed.pt"

    all_files = []

    for dir in args.indir:
        ds = SrcDataset(root=dir, ext=IMAGE_EXT)
        print(f"<{len(ds)}> files from {dir}")
        all_files += ds

    print(f"Total file count: {len(all_files)}")

    for i in tqdm(all_files, desc="Check images"):
        file_name, file_ext = os.path.splitext(i)

        embed_name = file_name + embed_file_tail

        if not os.path.exists(embed_name): 
            image_files.append(i)

    if len(image_files) == 0:
        print("No file to create embeddings")
        exit(0)

    persistent_workers=(os.name == 'nt')

    num_workers=args.num_workers

    if num_workers == 0:
        persistent_workers = False

    dataset = Dataset.from_dict({"image": image_files, "path": image_files }).cast_column("image", Image())
    dataset.set_transform(encode)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=num_workers, persistent_workers=persistent_workers)
    
    for batch in tqdm(dataloader, desc="Create embeddings"):
        pixel_values = batch["pixel_values"].to("cuda")
        clip_image_features = clip_model.get_image_features(pixel_values=pixel_values)

        for i, embedding in enumerate(clip_image_features):
            embed_file = batch['embed_file'][i]

            torch.save(embedding, embed_file)

        del pixel_values
        del batch
    