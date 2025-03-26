import os
import hashlib
import urllib
from tqdm import tqdm
import warnings
import numpy as np
import torch
from Masked_Clip import Masked_Clip


_MODELS = {
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
}


# 下载模型权重
def _download(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

# 构建模型
def load_model_checkpoint(args):
    if os.path.isfile(args.pretrained_vit):
        model_path = args.pretrained_vit
    else:
        model_path = _download(_MODELS[args.pretrained_vit])

    try:
        model = torch.jit.load(model_path, map_location="cpu")
        state_dict = None
    except RuntimeError:
        warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
        state_dict = torch.load(model_path, map_location="cpu")
    state_dict = state_dict or model.state_dict()

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size

    embed_dim = 512
    print('clip-vit layer num:', vision_layers)
    print('clip-vit width:', vision_width)
    print('clip-vit image_resolution:', image_resolution)
    print('clip-vit embed_dim:', embed_dim)

    model = Masked_Clip(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        args.text_encoder_width,args.image_mask,args.text_mask
    )
    print(model.state_dict().keys())
    #print(state_dict.keys())

    img_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('visual.'):
            img_state_dict['visual.' + k] = v
    img_state_dict['visual.visual.proj']
    msg = model.load_state_dict(img_state_dict, strict=False)
    #print(msg)

    return model


