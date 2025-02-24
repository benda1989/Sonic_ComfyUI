import os
import io
import time

import gc
import torch
import numpy as np
import folder_paths
import torchaudio

from omegaconf import OmegaConf
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import WhisperModel, AutoFeatureExtractor

from .sonic import Sonic, sonic_predata, preprocess_face, crop_face_image
from .src.dataset.test_preprocess import image_audio_to_tensor
from .src.models.audio_adapter.audio_proj import AudioProjModel
from .src.models.audio_adapter.audio_to_bucket import Audio2bucketModel
from .utils import tensor2cv, cv2pil,convert_cf2diffuser,tensor_upscale,tensor2pil
from .src.dataset.face_align.align import AlignImage


MAX_SEED = np.iinfo(np.int32).max
now_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(folder_paths.models_dir, "sonic")
os.makedirs(model_path,exist_ok=True)
folder_paths.add_model_folder_path("sonic", model_path)
assert torch.cuda.is_available(),"GKK·Sonic: Must have cuda"
device = torch.device( "cuda")

class Loader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "sonic_unet": (["none"] + folder_paths.get_filename_list("sonic"),),
                "ip_audio_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "use_interframe": ("BOOLEAN", {"default": True},),
                "dtype": (["fp16", "fp32", "bf16"],),
            },
        }

    RETURN_TYPES = ("MODEL_SONIC",)
    RETURN_NAMES = ("model",)
    FUNCTION = "run"
    CATEGORY = "GKK·Sonic"

    def run(self, model, sonic_unet, ip_audio_scale, use_interframe, dtype):
        print("GKK·Sonic: Start load model")
        assert sonic_unet is not None , 'GKK·Sonic: Loader need sonic_unet input'
        if dtype == "fp16":
            weight_dtype = torch.float16
        elif dtype == "fp32":
            weight_dtype = torch.float32
        elif dtype == "bf16":
            weight_dtype = torch.bfloat16
        svd_repo = os.path.join(now_dir, "svd_repo")
        val_scheduler = EulerDiscreteScheduler.from_pretrained(
            svd_repo,
            subfolder="scheduler")
        unet_file=os.path.join(svd_repo, "unet")
        unet=convert_cf2diffuser(model.model,unet_file,weight_dtype)
        vae_config=OmegaConf.load(os.path.join(svd_repo, "vae/config.json"))
        pipe = Sonic(device, weight_dtype, vae_config, val_scheduler, unet, 
                     os.path.join(model_path, "RIFE"), 
                     folder_paths.get_full_path("sonic", sonic_unet),
                     use_interframe, ip_audio_scale)
        gc.collect()
        torch.cuda.empty_cache()
        return (pipe,)

class Simper:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_SONIC",),
                "clip_vision": ("CLIP_VISION",),
                "vae": ("VAE",),
                "audio": ("AUDIO",),
                "image": ("IMAGE",),
                "min_resolution": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "display": "number"}),
                "expand_ratio": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "inference_steps": ("INT", {"default": 19, "min": 1, "max": 1024, "step": 1, "display": "number"}),
                "dynamic_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 19.0, "min": 5.0, "max": 120.0, "step": 0.5}),
            }}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "fps")
    FUNCTION = "run"
    CATEGORY = "GKK·Sonic"
    
    def run(self, model, clip_vision, vae, audio, image, min_resolution, expand_ratio,seed,inference_steps,dynamic_scale, fps):
        config = OmegaConf.load(os.path.join(now_dir, 'sonic.yaml'))

        audio2token_ckpt = os.path.join(model_path, "audio2token.pth")
        audio2bucket_ckpt = os.path.join(model_path, "audio2bucket.pth")
        yolo_ckpt = os.path.join(model_path, "yoloface_v5m.pt")
        assert os.path.exists(audio2bucket_ckpt) and os.path.exists(audio2token_ckpt) and os.path.exists(yolo_ckpt), "Please download the model first"

        whisper_repo = os.path.join(model_path, "whisper-tiny")
        whisper = WhisperModel.from_pretrained(whisper_repo).to(device).eval()
        whisper.requires_grad_(False)
        feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_repo)

        audio2token_dict = torch.load(audio2token_ckpt, map_location="cpu")
        audio2bucket_dict = torch.load(audio2bucket_ckpt, map_location="cpu")
        audio2token = AudioProjModel(seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=1024,
                                     context_tokens=32).to(device)
        audio2bucket = Audio2bucketModel(seq_len=50, blocks=1, channels=384, clip_channels=1024, intermediate_dim=1024,
                                         output_dim=1, context_tokens=2).to(device)
        audio2token.load_state_dict(audio2token_dict, strict=True)
        audio2bucket.load_state_dict(audio2bucket_dict, strict=True)
        del audio2token_dict, audio2bucket_dict

        duration = audio["waveform"].squeeze(0).shape[1] / audio["sample_rate"]
        audio_path = os.path.join(folder_paths.get_input_directory(), f"audio_{time.time()}_temp.wav")
        buff = io.BytesIO()
        torchaudio.save(buff, audio["waveform"].squeeze(0), audio["sample_rate"], format="FLAC")
        with open(audio_path, 'wb') as f:
            f.write(buff.getbuffer())
        gc.collect()
        torch.cuda.empty_cache()

        face_det = AlignImage(device, det_path=yolo_ckpt)
        cv_image = tensor2cv(image)
        face_info = preprocess_face(cv_image, face_det, expand_ratio=expand_ratio)
        if face_info['face_num'] > 0:
            cv2pil(crop_face_image(cv_image, face_info['crop_bbox']))

        print(f"GKK·Sonic: Start tensor img+audio")
        test_data = image_audio_to_tensor(face_det, 
                                          feature_extractor, 
                                          duration, 
                                          audio_path, 
                                          tensor2pil(image),
                                          limit=MAX_SEED, 
                                          image_size=min_resolution, 
                                          area=config.area)
        for k, v in test_data.items():
            if isinstance(v, torch.Tensor):
                test_data[k] = v.unsqueeze(0).to(device).float()

        ref_img = test_data['ref_img']
        step = 2
        _, audio_tensor_list, uncond_audio_tensor_list, motion_buckets, image_embeddings = sonic_predata(
            whisper, 
            test_data['audio_feature'], 
            test_data['audio_len'], 
            step, 
            audio2bucket, 
            clip_vision, 
            audio2token, 
            ref_img, 
            image, 
            device)
        del clip_vision, face_det, whisper
        audio2bucket.to("cpu")
        audio2token.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

        print(f"GKK·Sonic: Start infer {fps}f-{duration}s")
        height, width = ref_img.shape[-2:]
        img_latent=vae.encode(tensor_upscale(image,width,height)).to(device, dtype=torch.float16)
        imgs = model.process(audio_tensor_list,
                              uncond_audio_tensor_list,
                              motion_buckets,
                              test_data,
                              config,
                              image_embeds=image_embeddings,
                              img_latent=img_latent,
                              fps=fps,
                              vae=vae,
                              inference_steps=inference_steps,
                              dynamic_scale=dynamic_scale,
                              seed=seed)
        gc.collect()
        torch.cuda.empty_cache()
        # print(imgs.shape)
        return (imgs.permute(0, 2, 3, 4, 1).squeeze(0), fps)

class Speechs(Simper):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_SONIC",),
                "clip_vision": ("CLIP_VISION",),
                "vae": ("VAE",),
                "speechs": ("speechs_dict",),
                "image": ("IMAGE",),
                "min_resolution": ("INT", {"default": 512, "min": 128, "max": 2048, "step": 64, "display": "number"}),
                "expand_ratio": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": MAX_SEED}),
                "inference_steps": ("INT", {"default": 15, "min": 1, "max": 1024, "step": 1, "display": "number"}),
                "dynamic_scale": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 15.0, "min": 5.0, "max": 120.0, "step": 0.5}),
            }}
    def run(self, model, clip_vision,vae, speechs, image, min_resolution, expand_ratio,seed,inference_steps,dynamic_scale, fps):
        res =[]
        for i,speech in enumerate(speechs["speechs"]):
            print(f"GKK·Sonic: Start speechs part {i+1}")
            imgs, _ = super().run(model, clip_vision,vae, {"waveform":speech.unsqueeze(0),"sample_rate":speechs["sample_rate"]}, image, min_resolution, expand_ratio,seed,inference_steps,dynamic_scale, fps)
            res.append(imgs)
            # image = imgs[-1].unsqueeze(0)
        res = torch.cat(res, dim=0)
        print(res.shape)
        return (res, fps)
 

NODE_CLASS_MAPPINGS = {
    "SonicLoader": Loader,
    "SonicSimper": Simper,
    "SonicSpeechs": Speechs,
}

__all__ = ['NODE_CLASS_MAPPINGS']