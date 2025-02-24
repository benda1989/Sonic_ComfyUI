import os
import gc
import cv2
import torch
import numpy as np
from PIL import Image

from comfy.utils import common_upscale

cur_path = os.path.dirname(os.path.abspath(__file__))
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

def cv2pil(cv_image):
    rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

def convert_cf2diffuser(model,unet_config_file,weight_dtype):
    from .src.models.base.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
    cf_state_dict = model.diffusion_model.state_dict()
    unet_state_dict = model.model_config.process_unet_state_dict_for_saving(cf_state_dict)
    unet_config = UNetSpatioTemporalConditionModel.load_config(unet_config_file)
    Unet = UNetSpatioTemporalConditionModel.from_config(unet_config).to(device, weight_dtype)
    Unet.load_state_dict(unet_state_dict, strict=False)
    del cf_state_dict
    gc.collect()
    torch.cuda.empty_cache()
    return Unet

def tensor_to_pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    return Image.fromarray(image_np, mode='RGB')

def tensor_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    return img.movedim(1, -1)

def tensor2pil_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "nearest-exact", "center")
    samples = img.movedim(1, -1)
    return tensor_to_pil(samples)

def tensor2cv(tensor_image, RGB2BGR=True):
    if len(tensor_image.shape)==4:#bhwc to hwc
        tensor_image=tensor_image.squeeze(0)
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu().detach()
    tensor_image=tensor_image.numpy()
    maxValue=tensor_image.max()
    tensor_image=tensor_image*255/maxValue
    img_cv2=np.uint8(tensor_image)#32 to uint8
    if RGB2BGR:
        img_cv2=cv2.cvtColor(img_cv2,cv2.COLOR_RGB2BGR)
    return img_cv2

def tensor2pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image
