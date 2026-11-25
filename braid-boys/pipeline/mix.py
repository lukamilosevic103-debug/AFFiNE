#!/usr/bin/env python3
"""Final song mix: band + dad/son vocals + crowd + pyro FX, mastered for TikTok.
usage: mix.py CFG.json
"""
import sys, json, numpy as np, soundfile as sf, librosa, pyloudnorm as pyln
from scipy.signal import butter, sosfilt
from pedalboard import Pedalboard, Reverb, Compressor, Limiter, HighpassFilter, LowShelfFilter, PeakFilter, Gain

SR = 48000
rng = np.random.default_rng(5)


def load(p, n):
    y, sr = librosa.load(p, sr=SR, mono=False)
    if y.ndim == 1:
        y = np.stack([y, y])
    y = y.T
    out = np.zeros((n, 2), np.float32)
    m = min(n, len(y))
    out[:m] = y[:m]
    return out


def bp(x, lo, hi, order=2):
    return sosfilt(butter(order, [lo, hi], btype='band', fs=SR, output='sos'), x, axis=0)


def whoosh(dur=1.1):
    n = int(dur * SR)
    t = np.arange(n) / SR
    env = np.minimum(1, t / 0.03) * np.exp(-t / 0.35)
    x = rng.standard_normal((n, 2))
    y = bp(x, 120, 2600) * 0.8 + bp(x, 35, 140, 2) * 1.6
    # crackle
    cr = (rng.random((n, 2)) < 0.002) * rng.standard_normal((n, 2)) * 3
    y += bp(cr, 1500, 6000)
    return (y * env[:, None]).astype(np.float32)


def boom(dur=1.4):
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = 30 + 45 * np.exp(-t / 0.18)
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * np.exp(-t / 0.45)
    click = bp(rng.standard_normal(n) * np.exp(-t / 0.01), 800, 5000) * 0.4
    y = y + click
    return np.stack([y, y], 1).astype(np.float32)


def hey_chant(times):
    """crowd shouting HEY! on given times"""
    n_total = int((max(times) + 1.0) * SR)
    out = np.zeros((n_total, 2), np.float32)
    for tt in times:
        for v in range(36):
            dur = rng.uniform(0.18, 0.3)
            n = int(dur * SR)
            t = np.arange(n) / SR
            f0 = rng.uniform(130, 230) if rng.random() < 0.6 else rng.uniform(240, 420)
            f = f0 * (1 + 0.15 * np.exp(-t / 0.05)) * (1 - 0.1 * t / dur)
            src = 2 * ((np.cumsum(f) / SR) % 1) - 1 + 0.3 * rng.standard_normal(n)
            y = np.zeros(n)
            for fc, bw, g in ((530, 140, 1.0), (1850, 200, 0.7), (2500, 260, 0.35)):
                y += g * sosfilt(butter(2, [fc - bw, fc + bw], btype='band', fs=SR, output='sos'), src)
            hh = np.zeros(n)
            hh[:int(0.04 * SR)] = rng.standard_normal(int(0.04 * SR)) * 0.5  # the 'h'
            y += sosfilt(butter(2, [1200, 5000], btype='band', fs=SR, output='sos'), hh)
            env = np.minimum(1, t / 0.015) * np.clip((dur - t) / 0.08, 0, 1)
            y *= env * rng.uniform(0.4, 1.0)
            st = max(0, int((tt + 0.03 + rng.normal(0, 0.018)) * SR))
            pan = rng.uniform(0.1, 0.9)
            e = min(n_total, st + n)
            out[st:e, 0] += y[:e - st] * np.sqrt(1 - pan)
            out[st:e, 1] += y[:e - st] * np.sqrt(pan)
    out /= np.abs(out).max() + 1e-9
    return Pedalboard([Reverb(room_size=0.9, damping=0.4, wet_level=0.35, dry_level=0.8, width=1.0)])(out.T.copy(), SR).T


def add(dst, src, t, g=1.0):
    a = int(t * SR)
    if a >= len(dst):
        return
    b = min(len(dst), a + len(src))
    dst[a:b] += src[:b - a] * g


def main():
    cfg = json.load(open(sys.argv[1]))
    dur = cfg['duration']
    n = int(dur * SR)
    inst = load(cfg['inst'], n)
    dad = load(cfg['vox_dad'], n)
    son = load(cfg['vox_son'], n)
    crowd = load(cfg['crowd'], n)
    t = np.arange(n) / SR
    # band: live arena feel
    inst = Pedalboard([HighpassFilter(30), LowShelfFilter(cutoff_frequency_hz=90, gain_db=2.0),
                       Reverb(room_size=0.85, damping=0.5, wet_level=0.12, dry_level=0.95, width=1.0)])(inst.T.copy(), SR).T
    inst /= np.abs(inst).max() + 1e-9
    # duck band under vocals (smoothed sidechain)
    v = np.abs(dad).mean(1) + np.abs(son).mean(1)
    k = int(0.08 * SR)
    venv = np.convolve(v, np.ones(k) / k, mode='same')
    venv /= np.percentile(venv, 99) + 1e-9
    duck = 1 - 0.36 * np.clip(venv, 0, 1)
    # music ending: final hit then let the crowd take over
    end_fade = np.clip((cfg['music_end'] + 1.1 - t) / 1.1, 0, 1)
    band = inst * (duck * end_fade)[:, None] * 0.62
    vox = dad * cfg.get('dad_gain', 1.0) + son * cfg.get('son_gain', 1.0)

    def rms(x):
        return float(np.sqrt((x ** 2).mean()) + 1e-9)

    act = venv > 0.15
    vox *= rms(band[act]) / rms(vox[act]) * 10 ** (cfg.get('vox_db', 1.5) / 20)
    cr_gain = np.interp(t, *zip(*cfg['crowd_gain']))
    crowd = crowd * cr_gain[:, None]
    br = (t > cfg['break'][0]) & (t < cfg['break'][1])
    crowd *= rms(band[br]) / rms(crowd[br]) * 10 ** (cfg.get('crowd_break_db', -3.0) / 20)
    print('levels (dBFS rms): band', round(20 * np.log10(rms(band)), 1), 'vox(active)', round(20 * np.log10(rms(vox[act])), 1),
          'crowd(break)', round(20 * np.log10(rms(crowd[br])), 1), 'crowd(under vox)', round(20 * np.log10(rms(crowd[act])), 1))
    mix = band + vox + crowd
    fx = np.zeros_like(mix)
    for h in cfg['pyro']:
        add(fx, whoosh(), h, 0.22)
    for h in cfg['booms']:
        add(fx, boom(), h, 0.55)
    if cfg.get('hey'):
        add(fx, hey_chant([x - cfg['hey'][0] for x in cfg['hey']]), cfg['hey'][0] - 0.03, 0.35)
    mix += fx
    # master
    mix = Pedalboard([Compressor(threshold_db=-16, ratio=2.5, attack_ms=12, release_ms=140),
                      PeakFilter(cutoff_frequency_hz=3500, gain_db=1.0, q=0.7)])(mix.T.copy().astype(np.float32), SR).T
    meter = pyln.Meter(SR)
    lufs = meter.integrated_loudness(mix)
    mix = mix * 10 ** ((cfg.get('lufs', -10.5) - lufs) / 20)
    mix = Pedalboard([Limiter(threshold_db=-1.2, release_ms=80)])(mix.T.copy().astype(np.float32), SR).T
    # the JUCE limiter adds make-up gain: re-normalise loudness after it, keep >= 1 dB headroom
    lufs2 = meter.integrated_loudness(mix)
    mix = mix * 10 ** ((cfg.get('lufs', -10.0) - lufs2) / 20)
    pk = float(np.abs(mix).max())
    if pk > 0.89:
        mix = mix * (0.89 / pk)
    # fades
    fi = int(0.02 * SR)
    mix[:fi] *= np.linspace(0, 1, fi)[:, None]
    fo = int(0.45 * SR)
    mix[-fo:] *= np.linspace(1, 0, fo)[:, None] ** 1.5
    sf.write(cfg['out'], mix, SR, subtype='PCM_24')
    print('final LUFS', round(meter.integrated_loudness(mix), 2), 'peak', round(float(np.abs(mix).max()), 3))


if __name__ == '__main__':
    main()
