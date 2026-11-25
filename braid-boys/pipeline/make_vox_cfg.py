#!/usr/bin/env python3
"""Song arrangement: which take sings which line, on which beat, on which notes.
usage: make_vox_cfg.py BEATS.json VOX_DIR OUT_DIR TAKES.json > cfg.json
TAKES.json: {"dad1": "dad1_sing_a.wav", "dad2": ..., "son1": ..., "son2": ..., "dad3": ..., "dad4": ...}
Beat 0 = first tracked beat. Drop at beat 13, heavier section from beat 39.
"""
import sys, json

beats = json.load(open(sys.argv[1]))['beats']
vox, out = sys.argv[2], sys.argv[3]
takes = json.load(open(sys.argv[4]))


def W(w, b, d, *n, show=None):
    o = dict(w=w, b=b, d=d, n=list(n))
    if show:
        o['show'] = show
    return o


# Dad melody sits low (F#2..F#3) near his natural pitch: deep and heavy.
DAD1 = [W('I', 0, .5, 42), W('FOUGHT', .5, .75, 47), W('THROUGH', 1.25, .75, 47), W('HELL', 2, 1.25, 49, 47, show='HELL,'),
        W('I', 3.5, .5, 42), W('CONQUERED', 4, 1, 47, 49), W('THE', 5, .5, 47), W('FLAME', 5.5, 1, 52, 47, show='FLAME!')]
DAD2 = [W('NOW', 0, .5, 42), W('RISE', .5, 1, 47, 52, show='RISE,'), W('MY', 1.5, .5, 49), W('SON', 2, 1.25, 52, 47, show='SON,'),
        W('AND', 3.5, .5, 42), W('CARRY', 4, 1, 47, 49), W('MY', 5, .5, 49), W('NAME', 5.5, 1.1, 54, 47, show='NAME!')]
# Son chorus: bright kid range (B3..B4)
SON1 = [W('I', 0, .5, 59), W('FEEL', .5, 1, 64), W('YOUR', 1.5, .5, 64), W('FIRE', 2, 1.5, 66, 64),
        W('RUNNING', 4, 1, 64), W('THROUGH', 5, 1, 66), W('ME', 6, 1.75, 64, 59, show='ME!')]
SON2 = [W('THE', 0, .5, 59), W('BEAST', .5, 1, 64, 66), W('YOU', 1.5, .5, 64), W('ARE', 2, 1.5, 69, 66, show='ARE,'),
        W('IS', 4, .5, 64), W('THE', 4.5, .5, 64), W('BEAST', 5, 1, 71, 69), W("I'LL", 6, .75, 66), W('BE', 6.75, 1.75, 64, show='BE!')]
TAG = [W('THE', 0, .5, 59), W('BEAST', .5, 1.5, 71, 69), W("I'LL", 2, 1, 66), W('BE', 3, 3, 64, show='BE!')]


def oct_down(words, k=12):
    return [dict(w, n=[x - k for x in w['n']]) for w in words]


lines = [
    dict(id='verse1', singer='dad', take=f"{vox}/{takes['dad1']}", start_beat=0, words=DAD1, formant=0.93, keep_natural=0.3),
    dict(id='verse2', singer='dad', take=f"{vox}/{takes['dad2']}", start_beat=6.5, words=DAD2, formant=0.93, keep_natural=0.3),
    dict(id='chorus1a', singer='son', take=f"{vox}/{takes['son1']}", start_beat=13, words=SON1, formant=takes.get('son_formant', 1.1)),
    dict(id='chorus1b', singer='son', take=f"{vox}/{takes['son2']}", start_beat=21, words=SON2, formant=takes.get('son_formant', 1.1)),
    # chorus 2: son lead + dad an octave below
    dict(id='chorus2a', singer='son', take=f"{vox}/{takes['son1']}", start_beat=39, words=SON1, formant=takes.get('son_formant', 1.1), sub_singer='both'),
    dict(id='chorus2a_dad', singer='dad', take=f"{vox}/{takes['dad3']}", start_beat=39, words=oct_down(SON1), formant=0.93, gain=0.8, subtitle=False),
    dict(id='chorus2b', singer='son', take=f"{vox}/{takes['son2']}", start_beat=47, words=SON2, formant=takes.get('son_formant', 1.1), sub_singer='both'),
    dict(id='chorus2b_dad', singer='dad', take=f"{vox}/{takes['dad4']}", start_beat=47, words=oct_down(SON2), formant=0.93, gain=0.8, subtitle=False),
    # outro tag
    dict(id='tag', singer='son', take=f"{vox}/{takes['son2']}", start_beat=56, words=TAG, formant=takes.get('son_formant', 1.1), sub_singer='both', heard_skip=5),
    dict(id='tag_dad', singer='dad', take=f"{vox}/{takes['dad4']}", start_beat=56, words=oct_down(TAG), formant=0.93, gain=0.85, subtitle=False, heard_skip=6),
]
print(json.dumps(dict(beats=beats, duration=31.0, fps=30, out_dir=out, lines=lines), indent=1))
