import sys, time, json, torch
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler, AutoencoderKL
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 4)
jobs = json.load(open(sys.argv[1])); out = sys.argv[2]
base = 'stabilityai/stable-diffusion-xl-base-1.0'
dt = torch.bfloat16
unet = UNet2DConditionModel.from_config(base, subfolder='unet').to(dt)
unet.load_state_dict(load_file(hf_hub_download('ByteDance/SDXL-Lightning', 'sdxl_lightning_4step_unet.safetensors')))
kw = dict(torch_dtype=dt, variant='fp16')
pipe = StableDiffusionXLPipeline(
    vae=AutoencoderKL.from_pretrained(base, subfolder='vae', **kw),
    text_encoder=CLIPTextModel.from_pretrained(base, subfolder='text_encoder', **kw),
    text_encoder_2=CLIPTextModelWithProjection.from_pretrained(base, subfolder='text_encoder_2', **kw),
    tokenizer=CLIPTokenizer.from_pretrained(base, subfolder='tokenizer'),
    tokenizer_2=CLIPTokenizer.from_pretrained(base, subfolder='tokenizer_2'),
    unet=unet,
    scheduler=EulerDiscreteScheduler.from_pretrained(base, subfolder='scheduler', timestep_spacing='trailing'))
pipe.set_progress_bar_config(disable=True)
for j in jobs:
    t = time.time()
    g = torch.Generator().manual_seed(j['seed'])
    with torch.inference_mode():
        im = pipe(j['prompt'], num_inference_steps=4, guidance_scale=0, width=j.get('w', 768), height=j.get('h', 1344), generator=g).images[0]
    im.save(f"{out}/{j['name']}.png")
    print(j['name'], round(time.time() - t, 1), 's', flush=True)
