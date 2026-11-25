import sys, json, time, torch, numpy as np, soundfile as sf
from transformers import AutoProcessor, BarkModel
jobs = json.load(open(sys.argv[1])); out = sys.argv[2]; nth = int(sys.argv[3]) if len(sys.argv) > 3 else 4
torch.set_num_threads(nth)
proc = AutoProcessor.from_pretrained('suno/bark')
model = BarkModel.from_pretrained('suno/bark').eval()
sr = model.generation_config.sample_rate
for j in jobs:
    t = time.time()
    torch.manual_seed(j['seed'])
    inp = proc(j['text'], voice_preset=j['preset'])
    with torch.inference_mode():
        a = model.generate(**inp, do_sample=True, semantic_max_new_tokens=j.get('maxtok', 400), semantic_temperature=j.get('temp', 0.7), coarse_temperature=j.get('temp', 0.7), fine_temperature=0.5)
    a = a[0].float().numpy()
    sf.write(f"{out}/{j['name']}.wav", a, sr)
    print(j['name'], round(len(a)/sr, 2), 's', 'took', round(time.time()-t, 1), flush=True)
