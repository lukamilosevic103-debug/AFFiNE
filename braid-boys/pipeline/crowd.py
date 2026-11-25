#!/usr/bin/env python3
"""Synthesize a stereo arena crowd: roar bed, cheering 'woo' voices, applause, screams.
usage: crowd.py OUT.wav DURATION INTENSITY_JSON
INTENSITY_JSON: list of [t, level] breakpoints (level 0..1) for the overall crowd energy.
"""
import sys, json, numpy as np, soundfile as sf
from scipy.signal import butter, sosfilt, lfilter

SR = 48000
rng = np.random.default_rng(77)


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], btype='band', fs=SR, output='sos'), x)


def formant(x, freqs, bws, gains):
    y = np.zeros_like(x)
    for f, b, g in zip(freqs, bws, gains):
        y += g * bp(x, max(40, f - b / 2), f + b / 2, 2)
    return y


def envelope_curve(points, n):
    t = np.arange(n) / SR
    pts = np.array(points, float)
    return np.interp(t, pts[:, 0], pts[:, 1])


def roar(n):
    out = np.zeros((n, 2))
    for ch in range(2):
        w = rng.standard_normal(n)
        # pinkish
        w = lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400], w)
        v = formant(w, [650, 1150, 2600], [500, 700, 1200], [1.0, 0.7, 0.35])
        lfo = 0.75 + 0.25 * np.interp(np.arange(n), np.linspace(0, n, 40), rng.uniform(0, 1, 40))
        out[:, ch] = v * lfo
    return out / np.abs(out).max()


def voice(dur, f0a, f0b, f0c, vowel_a, vowel_b):
    m = int(dur * SR)
    t = np.arange(m) / SR
    # pitch glide a -> b -> c
    f0 = np.interp(t, [0, dur * 0.35, dur], [f0a, f0b, f0c]) * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * t))
    ph = np.cumsum(f0) / SR
    src = 2 * (ph % 1) - 1  # sawtooth
    src += 0.25 * rng.standard_normal(m)  # breath/rasp
    fa = np.array(vowel_a, float)
    fb = np.array(vowel_b, float)
    y = np.zeros(m)
    # formant filter in 4 chunks interpolating vowel
    k = 4
    edges = np.linspace(0, m, k + 1).astype(int)
    for i in range(k):
        u = i / (k - 1)
        fr = fa + (fb - fa) * u
        seg = src[edges[i]:edges[i + 1]]
        if len(seg) < 64:
            continue
        y[edges[i]:edges[i + 1]] = formant(seg, fr, [120, 160, 220], [1.0, 0.6, 0.25])
    env = np.minimum(1, t / 0.06) * np.minimum(1, (dur - t) / 0.25) ** 1.5
    return y * env


def cheer_layer(n, count, intensity):
    out = np.zeros((n, 2))
    U = (300, 870, 2240)
    O = (500, 900, 2400)
    A = (750, 1200, 2600)
    E = (530, 1850, 2500)
    for _ in range(count):
        # sample start time weighted by intensity
        for _try in range(10):
            st = rng.integers(0, n - SR // 2)
            if rng.random() < intensity[st] ** 1.5:
                break
        kind = rng.random()
        if kind < 0.45:   # high woo (women/kids)
            f = rng.uniform(330, 560)
            v = voice(rng.uniform(0.5, 1.3), f * 0.8, f * 1.25, f * 0.9, U, O)
        elif kind < 0.85:  # male yeah/woo
            f = rng.uniform(140, 260)
            v = voice(rng.uniform(0.5, 1.2), f * 0.85, f * 1.2, f * 0.8, rng.choice([0, 1]) and E or U, A)
        else:              # scream
            f = rng.uniform(650, 1000)
            v = voice(rng.uniform(0.4, 0.9), f, f * 1.1, f * 0.85, A, E)
        v *= rng.uniform(0.2, 1.0) / (np.abs(v).max() + 1e-9)
        pan = rng.uniform(0.1, 0.9)
        e = min(n, st + len(v))
        out[st:e, 0] += v[:e - st] * np.sqrt(1 - pan)
        out[st:e, 1] += v[:e - st] * np.sqrt(pan)
    return out


def applause(n, clappers, intensity):
    out = np.zeros((n, 2))
    clap = None
    for _ in range(clappers):
        rate = rng.uniform(3.5, 6.0)
        t = rng.uniform(0, 0.3)
        pan = rng.uniform(0, 1)
        fc = rng.uniform(900, 2800)
        L = int(SR * 0.012)
        burst = bp(rng.standard_normal(L) * np.exp(-np.arange(L) / (SR * 0.0025)), fc * 0.6, fc * 1.6)
        while t < n / SR - 0.05:
            i = int(t * SR)
            a = intensity[i] * rng.uniform(0.4, 1.0)
            if rng.random() < intensity[i] ** 0.7:
                out[i:i + L, 0] += burst * a * np.sqrt(1 - pan)
                out[i:i + L, 1] += burst * a * np.sqrt(pan)
            t += 1 / rate * rng.uniform(0.85, 1.15)
    return out


def simple_reverb(x, rt=2.2, mix=0.35):
    L = int(SR * rt)
    t = np.arange(L) / SR
    out = np.zeros_like(x)
    from scipy.signal import fftconvolve
    for ch in range(x.shape[1]):
        ir = rng.standard_normal(L) * np.exp(-6.9 * t / rt)
        ir = bp(ir, 200, 7000)
        ir /= np.sqrt((ir ** 2).sum())
        out[:, ch] = fftconvolve(x[:, ch], ir)[:len(x)]
    return x * (1 - mix) + out * mix * 3


def main():
    outp, dur = sys.argv[1], float(sys.argv[2])
    pts = json.loads(open(sys.argv[3]).read())
    n = int(dur * SR)
    I = envelope_curve(pts, n)
    r = roar(n) * (0.25 + 0.75 * I[:, None])
    c = cheer_layer(n, int(dur * 26), I)
    c /= np.abs(c).max() + 1e-9
    a = applause(n, 140, I)
    a /= np.abs(a).max() + 1e-9
    mix = r * 0.55 + c * 0.75 + a * 0.45
    mix = simple_reverb(mix, 2.4, 0.4)
    mix *= (0.15 + 0.85 * I[:, None])
    mix /= np.abs(mix).max() + 1e-9
    sf.write(outp, (mix * 0.9).astype(np.float32), SR)
    print('crowd written', outp, mix.shape)


if __name__ == '__main__':
    main()
