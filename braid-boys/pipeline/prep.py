# Precompute static layers for the compositor (photo space and 2x photo space)
import numpy as np, cv2, json, sys
from PIL import Image, ImageFilter
D = sys.argv[1]
photo = np.asarray(Image.open(f'{D}/photo.png').convert('RGB'))
H0, W0 = photo.shape[:2]
seg = np.load(f'{D}/seg.npy')
son = (seg == 1)
dad = (seg == 8) | (seg == 6)          # dad + microphone
# clean masks: fill holes, smooth
def clean(m):
    m = m.astype(np.uint8)*255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7,7),np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    big = 1+np.argmax(st[1:,cv2.CC_STAT_AREA])
    keep[lab==big] = 255
    # fill interior holes
    inv = cv2.bitwise_not(keep); n2, lab2, st2, _ = cv2.connectedComponentsWithStats(inv)
    for i in range(1,n2):
        if st2[i,cv2.CC_STAT_AREA] < 4000: keep[lab2==i] = 255
    return keep
son_m, dad_m = clean(son), clean(dad)
people = np.maximum(son_m, dad_m)
# background plate: inpaint people (dilated) so small layer motions reveal plausible pixels
dil = cv2.dilate(people, np.ones((15,15),np.uint8))
small = cv2.resize(photo, (W0//2, H0//2), interpolation=cv2.INTER_AREA)
sm = cv2.resize(dil, (W0//2, H0//2), interpolation=cv2.INTER_NEAREST)
bg_s = cv2.inpaint(cv2.cvtColor(small, cv2.COLOR_RGB2BGR), sm, 9, cv2.INPAINT_TELEA)
bg_s = cv2.cvtColor(bg_s, cv2.COLOR_BGR2RGB)
bg_full = cv2.resize(bg_s, (W0, H0), interpolation=cv2.INTER_CUBIC)
bg_full = cv2.GaussianBlur(bg_full, (0,0), 2.0)
md = (dil[...,None]/255.0)
bg = (photo*(1-md) + bg_full*md).astype(np.uint8)
# 2x upscale (Lanczos + gentle unsharp) for sharper close-ups
def up2(a, sharpen=True):
    u = cv2.resize(a, (W0*2, H0*2), interpolation=cv2.INTER_LANCZOS4)
    if sharpen and u.ndim == 3 and u.shape[2] == 3:
        b = cv2.GaussianBlur(u, (0,0), 1.6)
        u = cv2.addWeighted(u, 1.55, b, -0.55, 0)
    return u
photo2 = up2(photo); bg2 = up2(bg)
def feather(m):
    m2 = cv2.resize(m, (W0*2, H0*2), interpolation=cv2.INTER_LINEAR)
    m2 = cv2.GaussianBlur(m2, (0,0), 1.4)
    return m2
son_a2, dad_a2 = feather(son_m), feather(dad_m)
np.save(f'{D}/photo2.npy', photo2); np.save(f'{D}/bg2.npy', bg2)
np.save(f'{D}/son_a2.npy', son_a2); np.save(f'{D}/dad_a2.npy', dad_a2)
# flame mask (photo space): very bright warm pixels outside people, upper 60% of frame
hsv = cv2.cvtColor(photo, cv2.COLOR_RGB2HSV).astype(float)
v = hsv[...,2]/255; s = hsv[...,1]/255
lum = photo.mean(axis=2)/255
fm = ((v > 0.80) & (lum > 0.62)).astype(np.uint8)
fm[int(H0*0.62):] = 0
fm[people > 0] = 0
fm = cv2.morphologyEx(fm, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
n, lab, st, cen = cv2.connectedComponentsWithStats(fm)
cols = []
flame = np.zeros_like(fm)
for i in range(1, n):
    x,y,w,h,a = st[i]
    if a > 6000 and h > 200:
        flame[lab==i] = 1
        cols.append(dict(x=int(x),y=int(y),w=int(w),h=int(h),area=int(a),cx=float(cen[i][0])))
flame_soft = cv2.GaussianBlur(cv2.dilate(flame*255, np.ones((21,21),np.uint8)), (0,0), 12)
np.save(f'{D}/flame_soft.npy', flame_soft)
# stage lamps: small bright blobs in the truss band
band = (lum > 0.93).astype(np.uint8); band[:330] = 0; band[1000:] = 0
band[flame_soft > 40] = 0; band[people > 0] = 0
n, lab, st, cen = cv2.connectedComponentsWithStats(band)
lamps = [dict(x=float(cen[i][0]), y=float(cen[i][1]), a=int(st[i][4])) for i in range(1,n) if 6 <= st[i][4] <= 900]
meta = dict(W0=W0, H0=H0, flames=cols, lamps=lamps,
            son_bbox=[int(v) for v in cv2.boundingRect(son_m)], dad_bbox=[int(v) for v in cv2.boundingRect(dad_m)])
json.dump(meta, open(f'{D}/meta.json','w'), indent=1)
print(json.dumps(dict(flames=cols, nlamps=len(lamps), son=meta['son_bbox'], dad=meta['dad_bbox'])))
# previews
pv = photo.astype(float).copy()
pv[flame>0] = pv[flame>0]*0.4 + np.array([0,255,255])*0.6
for l in lamps: cv2.circle(pv, (int(l['x']),int(l['y'])), 9, (255,0,0), 2)
Image.fromarray(np.clip(pv,0,255).astype(np.uint8)).resize((562,916)).save(f'{D}/prep_preview.jpg')
Image.fromarray(bg).resize((562,916)).save(f'{D}/bg_preview.jpg')
