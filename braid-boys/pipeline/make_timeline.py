#!/usr/bin/env python3
"""Build the video timeline (shots, pyro, subtitles, envelopes) from the song analysis.
usage: make_timeline.py BEATS.json VOX_DIR INST.wav OUT_TIMELINE.json
"""
import sys, json, numpy as np, librosa

beats_d = json.load(open(sys.argv[1]))
B_ = np.array(beats_d['beats'])
vox = sys.argv[2]
inst = sys.argv[3]
outp = sys.argv[4]
FPS, DUR = 30, 30.0


def B(b):
    if b <= 0:
        return float(B_[0] + b * (B_[1] - B_[0]))
    if b >= len(B_) - 1:
        return float(B_[-1] + (b - len(B_) + 1) * (B_[-1] - B_[-2]))
    i = int(np.floor(b))
    return float(B_[i] + (b - i) * (B_[i + 1] - B_[i]))


# --- envelopes
nfr = int(DUR * FPS)
y, sr = librosa.load(inst, sr=22050, mono=True)
hop = sr // FPS
rms = np.array([np.sqrt((y[i * hop:(i + 1) * hop] ** 2).mean()) for i in range(nfr)])
rms = np.convolve(rms, np.ones(9) / 9, mode='same')
energy = np.clip((rms - np.percentile(rms, 5)) / (np.percentile(rms, 97) - np.percentile(rms, 5)), 0, 1)
env = dict(energy=energy.round(3).tolist())
for s in ('dad', 'son'):
    e = np.load(f'{vox}/env_{s}.npy')[:nfr]
    e = np.pad(e, (0, nfr - len(e)))
    env[s] = e.round(3).tolist()

# --- words / subtitle chunks
words = json.load(open(f'{vox}/words.json'))
chunks = []
lines = {}
for i, w in enumerate(words):
    lines.setdefault(w['line'], []).append(i)
SPLIT_AFTER = {'FIRE', 'HELL,', 'ARE,', 'SON,', 'RISE,'}
for lid, idx in sorted(lines.items(), key=lambda kv: words[kv[1][0]]['t0']):
    cur = []
    for i in idx:
        cur.append(i)
        if words[i]['text'] in SPLIT_AFTER and words[i]['text'] != 'RISE,':
            chunks.append(cur)
            cur = []
    if cur:
        chunks.append(cur)
ch = []
for k, c in enumerate(chunks):
    t0 = words[c[0]]['t0'] - 0.06
    t1 = words[c[-1]]['t1'] + 0.3
    ch.append(dict(t0=round(t0, 3), t1=round(t1, 3), words=c, singer=words[c[0]]['singer']))
for k in range(len(ch) - 1):
    ch[k]['t1'] = round(min(ch[k]['t1'], ch[k + 1]['t0'] - 0.01), 3)

# --- camera presets (photo coordinates)
WIDE = dict(cx=562, cy=916, vh=1790)
WIDE_T = dict(cx=560, cy=880, vh=1560)
TWO = dict(cx=575, cy=800, vh=1420)
TWO_T = dict(cx=575, cy=780, vh=1250)
DAD_MCU = dict(cx=838, cy=560, vh=900)
DAD_MCU2 = dict(cx=845, cy=520, vh=800)
DAD_CU = dict(cx=850, cy=440, vh=620)
DAD_CU2 = dict(cx=852, cy=425, vh=540)
SON_MCU = dict(cx=290, cy=1150, vh=840)
SON_MCU2 = dict(cx=288, cy=1110, vh=740)
SON_CU = dict(cx=285, cy=1045, vh=600)
SON_CU2 = dict(cx=285, cy=1040, vh=520)
LOW_TWO = dict(cx=560, cy=1000, vh=1500, rot=-3)
LOW_TWO2 = dict(cx=560, cy=960, vh=1350, rot=-1.5)


def S(t0, t1, a, b=None, **kw):
    d = dict(t0=round(t0, 3), t1=round(t1, 3), a=a, b=b or a)
    d.update(kw)
    return d


shots = [
    S(0.0, B(0), dict(WIDE, vh=1830), WIDE_T, fg_crowd=1.0, enter=0.12, ease='out'),
    S(B(0), B(3.5), DAD_MCU, DAD_MCU2, shake=5),
    S(B(3.5), B(6.5), DAD_CU, DAD_CU2, whip_in=True, shake=6),
    S(B(6.5), B(9.5), SON_MCU, SON_MCU2, whip_in=True, whip_dir=-1, shake=6),
    S(B(9.5), B(13), TWO, TWO_T, fg_crowd=0.8, ease='lin', shake=5),
    # DROP
    S(B(13), B(17), SON_CU, SON_CU2, enter=0.28, shake=9, punch=0.05),
    S(B(17), B(21), WIDE_T, dict(WIDE_T, vh=1450), fg_crowd=1.0, shake=10, punch=0.05),
    S(B(21), B(24), DAD_MCU, DAD_MCU2, whip_in=True, shake=8),
    S(B(24), B(29.5), SON_CU, SON_CU2, whip_in=True, whip_dir=-1, shake=9, punch=0.05),
    # BREAK: crowd goes wild
    S(B(29.5), B(32.5), dict(zoom=1.0, px=0.0, py=0.02), dict(zoom=1.1, px=0.0, py=0.0), kind='image', img='arena_wide_1', enter=0.2, shake=7, fg_crowd=0.6),
    S(B(32.5), B(35.5), dict(zoom=1.04, px=-0.02, py=0.0), dict(zoom=1.14, px=0.02, py=0.0), kind='image', img='front_row_1', shake=10),
    S(B(35.5), B(39), LOW_TWO, LOW_TWO2, fg_crowd=1.0, shake=8, ease='lin'),
    # CHORUS 2 (both)
    S(B(39), B(43), dict(WIDE, vh=1700), dict(WIDE, vh=1560), fg_crowd=1.0, enter=0.3, shake=12, punch=0.06),
    S(B(43), B(45), DAD_CU, DAD_CU2, shake=10, punch=0.05),
    S(B(45), B(47), SON_CU, SON_CU2, shake=10, punch=0.05),
    S(B(47), B(51), TWO, TWO_T, fg_crowd=0.8, whip_in=True, shake=10, punch=0.05),
    S(B(51), B(53), dict(zoom=1.0, px=0.0, py=0.0), dict(zoom=1.08, px=0.0, py=-0.02), kind='image', img='arena_wide_2', shake=8, fg_crowd=0.5),
    S(B(53), B(56), SON_MCU, SON_CU, shake=10, punch=0.05),
    # OUTRO tag + finale
    S(B(56), B(60), LOW_TWO, dict(LOW_TWO2, vh=1250), fg_crowd=0.9, shake=7, ease='io'),
    S(B(60), B(62), DAD_CU, DAD_CU2, shake=8),
    S(B(62), DUR + 0.1, dict(WIDE, vh=1450), dict(WIDE, vh=1830), fg_crowd=1.0, enter=0.25, shake=10, ease='out'),
]

big = [0.02, B(13), B(39), B(62)]
med = [B(2), B(5.5), B(15), B(21), B(26), B(29.5), B(31), B(33), B(35), B(37), B(43), B(47), B(51), B(56), B(58)]
hits = [dict(t=round(t, 3), a=1.0) for t in big] + [dict(t=round(t, 3), a=0.75) for t in med]
hits.sort(key=lambda h: h['t'])

strobes = [round(B(b), 3) for b in np.arange(33, 39, 0.5)] + [round(B(b), 3) for b in range(39, 56)]

glints = []
for w in words:
    if w['text'].startswith('BEAST') and w['singer'] in ('son', 'both'):
        glints.append(dict(t0=w['t0'] - 0.05, t1=w['t1'] + 0.25, who='son'))
# dad's eyes flare when the son sings "THE BEAST YOU ARE" at him (chorus 1)
glints.append(dict(t0=B(21.5) - 0.05, t1=B(22.5) + 0.2, who='dad'))

tl = dict(
    fps=FPS, duration=DUR, bpm=float(beats_d['tempo']), beats=[round(float(b), 3) for b in B_], hits=hits, strobes=strobes,
    env=env,
    jets=[dict(x=40, y=820, w=130, h=760), dict(x=325, y=880, w=190, h=840), dict(x=755, y=900, w=190, h=850), dict(x=1062, y=990, w=160, h=880)],
    faces=dict(son=dict(mouth=[285, 1062], chin=[285, 1094], rx=30, ry=22, amp=7, eyes=[[258, 1005], [312, 1005]]),
               dad=dict(mouth=[848, 414], chin=[856, 462], rx=34, ry=26, amp=6, block_x=862, block_y=445, eyes=[[826, 344], [886, 343]])),
    shots=shots, words=words, chunks=ch,
    logos=[dict(t0=0.0, t1=B(0) - 0.05, slam=True, y=700, tag='LIVE IN CONCERT'),
           dict(t0=B(62) + 0.05, t1=DUR + 0.1, slam=True, y=760, tag='FOLLOW FOR PART 2')],
    flashes=[dict(t=0.0, a=1.0, decay=0.12), dict(t=B(13), a=0.9), dict(t=B(29.5), a=0.6), dict(t=B(39), a=1.0), dict(t=B(56), a=0.5), dict(t=B(62), a=1.0, decay=0.12)],
    fireballs=[dict(t=B(13)), dict(t=B(39)), dict(t=B(62))],
    eye_glints=glints, sub_y=1215,
)
json.dump(tl, open(outp, 'w'))
print('shots', len(shots), 'chunks', len(ch), 'hits', len(hits), 'last word', words[-1])
for c in ch:
    print(f"  {c['t0']:6.2f}-{c['t1']:6.2f} [{c['singer']}] " + ' '.join(words[i]['text'] for i in c['words']))
