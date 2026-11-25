import sys, torch, numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
torch.set_num_threads(2)
src, out = sys.argv[1], sys.argv[2]
img = Image.open(src).convert('RGB')
mid = 'facebook/mask2former-swin-large-coco-panoptic'
proc = AutoImageProcessor.from_pretrained(mid)
model = Mask2FormerForUniversalSegmentation.from_pretrained(mid).eval()
inp = proc(images=img, return_tensors='pt')
with torch.inference_mode():
    o = model(**inp)
res = proc.post_process_panoptic_segmentation(o, target_sizes=[img.size[::-1]], threshold=0.5, label_ids_to_fuse=set())[0]
seg = res['segmentation'].numpy()
for s in res['segments_info']:
    lab = model.config.id2label[s['label_id']]
    m = seg == s['id']
    ys, xs = np.where(m)
    print(s['id'], lab, round(s['score'],3), m.sum(), (xs.min(), ys.min(), xs.max(), ys.max()))
np.save(out, seg)
