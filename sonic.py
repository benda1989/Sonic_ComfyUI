import os
import torch
import torch.utils.checkpoint
from tqdm import tqdm
import gc

from .src.utils.util import seed_everything
from .src.dataset.test_preprocess import process_bbox
from .src.models.base.unet_spatio_temporal_condition import  add_ip_adapters
from .src.pipelines.pipeline_sonic import SonicPipeline
from .src.utils.RIFE.RIFE_HDv3 import RIFEModel



BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def sonic_predata(wav_enc,audio_feature,audio_len,step,audio2bucket,image_encoder,audio_pe,ref_img,clip_img,device):
 
    image_embeds=image_encoder.encode_image(clip_img)["image_embeds"] #torch.Size([1, 1024])
    if device==torch.device("cuda"):
        image_embeds=image_embeds.clone().detach().to(device, dtype=torch.float16) #dtype需要改成可选fp16 bf16
    else:
        image_embeds=image_embeds.to(device, dtype=torch.float32) #dtype需要改成可选

    audio_prompts = []
    last_audio_prompts = []
    window = 3000
    #print(audio_feature.shape) #torch.Size([80, 3000])
    for i in range(0, audio_feature.shape[-1], window):
        audio_prompt = wav_enc.encoder(audio_feature[:,:,i:i+window], output_hidden_states=True).hidden_states
        last_audio_prompt = wav_enc.encoder(audio_feature[:,:,i:i+window]).last_hidden_state
        last_audio_prompt = last_audio_prompt.unsqueeze(-2)
        audio_prompt = torch.stack(audio_prompt, dim=2)
        audio_prompts.append(audio_prompt)
        last_audio_prompts.append(last_audio_prompt)

    audio_prompts = torch.cat(audio_prompts, dim=1)
    audio_prompts = audio_prompts[:,:audio_len*2]
    audio_prompts = torch.cat([torch.zeros_like(audio_prompts[:,:4]), audio_prompts, torch.zeros_like(audio_prompts[:,:6])], 1)

    last_audio_prompts = torch.cat(last_audio_prompts, dim=1)
    last_audio_prompts = last_audio_prompts[:,:audio_len*2]
    last_audio_prompts = torch.cat([torch.zeros_like(last_audio_prompts[:,:24]), last_audio_prompts, torch.zeros_like(last_audio_prompts[:,:26])], 1)


    ref_tensor_list = []
    audio_tensor_list = []
    uncond_audio_tensor_list = []
    motion_buckets = []
    for i in tqdm(range(audio_len//step)):
        audio_clip = audio_prompts[:,i*2*step:i*2*step+10].unsqueeze(0)
        audio_clip_for_bucket = last_audio_prompts[:,i*2*step:i*2*step+50].unsqueeze(0)
        motion_bucket = audio2bucket(audio_clip_for_bucket, image_embeds)
        motion_bucket = motion_bucket * 16 + 16
        motion_buckets.append(motion_bucket[0])

        cond_audio_clip = audio_pe(audio_clip).squeeze(0)
        uncond_audio_clip = audio_pe(torch.zeros_like(audio_clip)).squeeze(0)

        ref_tensor_list.append(ref_img[0])
        audio_tensor_list.append(cond_audio_clip[0])
        uncond_audio_tensor_list.append(uncond_audio_clip[0])

    return ref_tensor_list,audio_tensor_list,uncond_audio_tensor_list,motion_buckets,image_embeds



def preprocess_face(face_image,face_det, expand_ratio=1.0):
        
        h, w = face_image.shape[:2]
        _, _, bboxes = face_det(face_image, maxface=True)
        face_num = len(bboxes)
        bbox = []
        if face_num > 0:
            x1, y1, ww, hh = bboxes[0]
            x2, y2 = x1 + ww, y1 + hh
            bbox = x1, y1, x2, y2
            bbox_s = process_bbox(bbox, expand_radio=expand_ratio, height=h, width=w)

        return {
            'face_num': face_num,
            'crop_bbox': bbox_s,
        }

def crop_face_image(face_image,crop_bbox):
    crop_image = face_image[crop_bbox[1]:crop_bbox[3], crop_bbox[0]:crop_bbox[2]]
    return crop_image


def decode_latents_(latents, num_frames,vae, decode_chunk_size=14):
        # [batch, frames, channels, height, width] -> [batch*frames, channels, height, width]
        latents = latents.flatten(0, 1)

        latents = 1 / 0.18215 * latents
        if latents.is_cuda and vae.device==torch.device("cpu"):
            latents = latents.to(vae.device)
        # forward_vae_fn = self.vae._orig_mod.forward if is_compiled_module(self.vae) else self.vae.forward
        # accepts_num_frames = "num_frames" in set(inspect.signature(forward_vae_fn).parameters.keys())

        # decode decode_chunk_size frames at a time to avoid OOM
        frames = []
        for i in range(0, latents.shape[0], decode_chunk_size):
            #num_frames_in = latents[i : i + decode_chunk_size].shape[0]
            #decode_kwargs = {}
            # if accepts_num_frames:
            #     # we only pass num_frames_in if it's expected
            #     decode_kwargs["num_frames"] = num_frames_in

            frame = vae.decode(latents[i : i + decode_chunk_size])
            frames.append(frame.cpu())
        frames = torch.cat(frames, dim=0) # [50, 512, 512, 3]

        # [batch*frames, channels, height, width] -> [batch, channels, frames, height, width]
        #frames = frames.reshape(-1, num_frames, *frames.shape[1:]).permute(0, 2, 1, 3, 4)
        frames=frames.unsqueeze(0).permute(0, 4, 1, 2, 3)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        frames = frames.float()
        return frames



def test(
    pipe,
    config,
    audio_tensor_list,
    uncond_audio_tensor_list,
    motion_buckets,
    width,
    height,
    batch,
    image_embeds,
    fps,
    img_latent,
    vae,
):

    ref_img = batch['ref_img']
    face_mask= batch['face_mask']
    video = pipe(
        ref_img,
        image_embeds,
        face_mask,
        audio_tensor_list,
        uncond_audio_tensor_list,
        motion_buckets,
        height=height,
        width=width,
        num_frames=len(audio_tensor_list),
        decode_chunk_size=config.decode_chunk_size,
        motion_bucket_scale=config.motion_bucket_scale,
        fps=fps,
        noise_aug_strength=config.noise_aug_strength,
        min_guidance_scale1=config.min_appearance_guidance_scale, # 1.0,
        max_guidance_scale1=config.max_appearance_guidance_scale,
        min_guidance_scale2=config.audio_guidance_scale, # 1.0,
        max_guidance_scale2=config.audio_guidance_scale,
        output_type='latent',
        overlap=config.overlap,
        shift_offset=config.shift_offset,
        frames_per_batch=int(fps),
        num_inference_steps=config.num_inference_steps,
        i2i_noise_strength=config.i2i_noise_strength,
        img_latent=img_latent,
    ).frames

    
    video=decode_latents_(video, len(audio_tensor_list),vae, decode_chunk_size=14) # torch.Size([1, 3, 250, 512, 512])

    # Concat it with pose tensor
    # pose_tensor = torch.stack(pose_tensor_list,1).unsqueeze(0)
    # video = (video*0.5 + 0.5).clamp(0, 1)
    # video = torch.cat([video.to(pipe.device)], dim=0).cpu()

    return video





class Sonic():
   
    def __init__(self, 
                 device,
                 weight_dtype,
                 vae_config,
                 val_noise_scheduler,
                 unet,
                 flownet_ckpt,
                 sonic_unet,
                 use_interframe,
                 ip_audio_scale,
                 ):
        self.use_interframe = use_interframe
        #config.use_interframe = enable_interpolate_frame
        add_ip_adapters(unet, [32], [ip_audio_scale])
        sonic_dict = torch.load(sonic_unet, map_location="cpu")
        unet.load_state_dict(sonic_dict,strict=True,)
        del sonic_dict
        gc.collect()
        torch.cuda.empty_cache()

        if self.use_interframe:
            rife = RIFEModel(device=device)
            rife.load_model(flownet_ckpt)
            self.rife = rife

        #vae.to(weight_dtype)
        unet.to(weight_dtype)

        pipe = SonicPipeline(
            unet=unet,
            #vae=vae,
            scheduler=val_noise_scheduler,
            vae_config=vae_config,
        )
        self.pipe = pipe.to(dtype=weight_dtype)
        self.device = device
        print('init done')


    @torch.no_grad()
    def process(self,
                audio_tensor_list,
                uncond_audio_tensor_list,
                motion_buckets,
                test_data,
                config,
                image_embeds,
                img_latent,
                fps,
                vae,
                inference_steps=25,
                dynamic_scale=1.0,
                seed=None):
        
        # specific parameters
        if seed:
            config.seed = seed

        config.num_inference_steps = inference_steps
        config.motion_bucket_scale = dynamic_scale
        seed_everything(config.seed)
       
        height, width = test_data['ref_img'].shape[-2:]
        #self.pipe.enable_model_cpu_offload #太慢，没意义
        self.pipe.to(self.device)
        video = test(
            self.pipe,
            config,
            audio_tensor_list,
            uncond_audio_tensor_list,
            motion_buckets,
            width=width,
            height=height,
            batch=test_data,
            image_embeds=image_embeds,
            fps=fps,
            img_latent=img_latent,
            vae=vae,
            )

        if self.use_interframe:
            rife = self.rife
            out = video.to(self.device)
            results = []
            video_len = out.shape[2]
            for idx in tqdm(range(video_len-1), ncols=0):
                I1 = out[:, :, idx]
                I2 = out[:, :, idx+1]
                middle = rife.inference(I1, I2).clamp(0, 1).detach()
                results.append(out[:, :, idx])
                results.append(middle)
            results.append(out[:, :, video_len-1])
            video = torch.stack(results, 2).cpu()
         
        return video
        
