import sys, time, torch, numpy as np, soundfile as sf
from transformers import AutoProcessor, MusicgenForConditionalGeneration
model_id, seconds, seeds, outpref = sys.argv[1], float(sys.argv[2]), [int(s) for s in sys.argv[3].split(',')], sys.argv[4]
prompt = sys.argv[5]
torch.set_num_threads(4)
t=time.time()
proc = AutoProcessor.from_pretrained(model_id)
model = MusicgenForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16)
model.eval()
print('load', round(time.time()-t,1), flush=True)
sr = model.config.audio_encoder.sampling_rate
inputs = proc(text=[prompt], padding=True, return_tensors='pt')
toks = int(seconds*model.config.audio_encoder.frame_rate)
for s in seeds:
    torch.manual_seed(s)
    t=time.time()
    with torch.inference_mode():
        audio = model.generate(**inputs, do_sample=True, guidance_scale=3.5, max_new_tokens=toks)
    a = audio[0].float().cpu().numpy()  # (channels, samples)
    sf.write(f'{outpref}_{s}.wav', a.T, sr)
    print('seed', s, 'gen', round(time.time()-t,1), 's', a.shape, flush=True)
