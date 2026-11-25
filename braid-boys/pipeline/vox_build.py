#!/usr/bin/env python3
"""Turn raw Bark vocal takes into beat-locked, melody-tuned rock vocal stems.

usage: vox_build.py CONFIG.json
Writes per-singer stems (48 kHz stereo), word timings and per-frame vocal envelopes.
"""
import sys, json, re, difflib, numpy as np, soundfile as sf, pyworld as pw, librosa
from scipy.signal import resample_poly
from pedalboard import (Pedalboard, HighpassFilter, LowpassFilter, Compressor, Distortion, PeakFilter, Delay, Reverb, Gain,
                        Chorus, Limiter, LowShelfFilter)

FS = 24000
FP = 5.0  # WORLD frame period ms
OUT_SR = 48000


def norm(w):
    return re.sub(r"[^a-z']", '', w.lower())


def beat_time(beats, b):
    beats = np.asarray(beats)
    if b <= 0:
        return beats[0] + b * (beats[1] - beats[0])
    if b >= len(beats) - 1:
        return beats[-1] + (b - len(beats) + 1) * (beats[-1] - beats[-2])
    i = int(np.floor(b))
    return beats[i] + (b - i) * (beats[i + 1] - beats[i])


_whisper = None


def whisper_words(path):
    global _whisper
    from faster_whisper import WhisperModel
    if _whisper is None:
        _whisper = WhisperModel('small.en', device='cpu', compute_type='int8', cpu_threads=4)
    a16 = librosa.load(path, sr=16000, mono=True)[0].astype(np.float32)
    segs, _ = _whisper.transcribe(a16, word_timestamps=True, beam_size=5, language='en', vad_filter=False)
    out = []
    for s in segs:
        for w in s.words:
            for piece in w.word.strip().split():
                out.append(dict(w=piece, s=w.start, e=w.end, p=w.probability))
    return out


def align(target, heard):
    """Map each target word to a (start,end) in the take using sequence alignment; interpolate gaps."""
    tn = [norm(w) for w in target]
    hn = [norm(h['w']) for h in heard]
    sm = difflib.SequenceMatcher(a=tn, b=hn, autojunk=False)
    spans = [None] * len(target)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            spans[a + k] = [heard[b + k]['s'], heard[b + k]['e']]
    # fuzzy fill for near matches (e.g. 'conquerd')
    for i, sp in enumerate(spans):
        if sp is None:
            best, bj = 0, None
            for j, h in enumerate(hn):
                r = difflib.SequenceMatcher(a=tn[i], b=h).ratio()
                if r > best:
                    best, bj = r, j
            if best >= 0.6:
                spans[i] = [heard[bj]['s'], heard[bj]['e']]
    matched = sum(s is not None for s in spans)
    # interpolate missing
    known = [i for i, s in enumerate(spans) if s is not None]
    if not known:
        return None, 0.0
    for i in range(len(spans)):
        if spans[i] is None:
            prev = max([k for k in known if k < i], default=None)
            nxt = min([k for k in known if k > i], default=None)
            if prev is not None and nxt is not None:
                a, b = spans[prev][1], spans[nxt][0]
                gap = (b - a) / (nxt - prev)
                spans[i] = [a + gap * (i - prev - 1) + 0.01, a + gap * (i - prev) - 0.01]
            elif prev is not None:
                spans[i] = [spans[prev][1] + 0.02, spans[prev][1] + 0.3]
            else:
                spans[i] = [max(0, spans[nxt][0] - 0.3), spans[nxt][0] - 0.02]
    # enforce monotonic non-overlap
    for i in range(1, len(spans)):
        if spans[i][0] < spans[i - 1][1]:
            mid = (spans[i][0] + spans[i - 1][1]) / 2
            spans[i - 1][1] = mid
            spans[i][0] = mid
    return spans, matched / len(target)


def refine(spans, x):
    """Snap boundaries to energy: trim leading/trailing silence of each word."""
    hop = int(FS * 0.005)
    e = librosa.feature.rms(y=x, frame_length=hop * 4, hop_length=hop)[0]
    out = []
    for s, t in spans:
        a, b = int(s / 0.005), int(t / 0.005)
        a0, b0 = max(0, a - 16), min(len(e) - 1, b + 22)
        seg = e[a0:b0 + 1]
        if len(seg) == 0:
            out.append([s, t])
            continue
        # low thresholds so soft consonants (b-, f-, th-, -st, -t) are kept
        idx_s = np.where(seg > seg.max() * 0.06)[0]
        idx_e = np.where(seg > seg.max() * 0.025)[0]
        if len(idx_s) == 0 or len(idx_e) == 0:
            out.append([s, t])
            continue
        na, nb = a0 + idx_s[0] - 3, a0 + idx_e[-1] + 4
        na = max(na, a - 16)
        nb = min(nb, b + 22)
        out.append([max(0, na) * 0.005, (nb + 1) * 0.005])
    for i in range(1, len(out)):
        if out[i][0] < out[i - 1][1]:
            mid = (out[i][0] + out[i - 1][1]) / 2
            out[i - 1][1], out[i][0] = mid, mid
    return out


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def build_line(line, beats, cache):
    take = line['take']
    if take not in cache:
        x, sr = sf.read(take)
        if x.ndim > 1:
            x = x.mean(1)
        if sr != FS:
            x = librosa.resample(x, orig_sr=sr, target_sr=FS)
        x = x.astype(np.float64)
        f0, t = pw.harvest(x, FS, f0_floor=60, f0_ceil=900, frame_period=FP)
        sp = pw.cheaptrick(x, f0, t, FS)
        ap = pw.d4c(x, f0, t, FS)
        heard = whisper_words(take)
        cache[take] = (x, f0, sp, ap, heard)
    x, f0, sp, ap, heard = cache[take]
    words = [w['w'] for w in line['words']]
    spans, score = align(words, heard[line.get('heard_skip', 0):])
    spans = refine(spans, x.astype(np.float32))
    nsrc = len(f0)
    shift = line.get('semitones', 0)
    alpha = line.get('formant', 1.0)
    keep = line.get('keep_natural', 0.25)
    t0_line = beat_time(beats, line['start_beat'] + line['words'][0]['b']) - 0.03
    t1_line = beat_time(beats, line['start_beat'] + line['words'][-1]['b'] + line['words'][-1]['d'])
    nout = int((t1_line - t0_line) * 1000 / FP) + 40
    of0 = np.zeros(nout)
    osp = np.tile(sp.min(axis=0) * 1e-3, (nout, 1))
    oap = np.ones((nout, ap.shape[1]))
    gate = np.zeros(nout)
    timings = []
    for wi, (w, (ss, se)) in enumerate(zip(line['words'], spans)):
        ws = beat_time(beats, line['start_beat'] + w['b'])
        we = beat_time(beats, line['start_beat'] + w['b'] + w['d'])
        snd = (we - ws) * (0.94 if wi < len(line['words']) - 1 else 1.0)
        timings.append(dict(text=w.get('show', w['w']), t0=round(ws, 3), t1=round(ws + snd, 3)))
        src_len = max(0.04, se - ss)
        head = min(0.07, 0.3 * src_len)
        tail = min(0.09, 0.3 * src_len)
        mid_src = max(0.01, src_len - head - tail)
        mid_out = max(0.01, snd - head - tail)
        a = int((ws - t0_line) * 1000 / FP)
        b = int((ws + snd - t0_line) * 1000 / FP)
        notes = w['n']
        # natural reference pitch for this word
        sa, sb = int(ss * 1000 / FP), min(nsrc - 1, int(se * 1000 / FP))
        vf = f0[sa:sb + 1]
        vf = vf[vf > 0]
        med = np.median(vf) if len(vf) else 150.0
        for k in range(max(0, a), min(nout, b)):
            u = (k - a) * FP / 1000.0  # time into the word (s)
            if u < head:
                st = ss + u
            elif u > snd - tail:
                st = se - (snd - u)
            else:
                st = ss + head + (u - head) / mid_out * mid_src
            fi = min(nsrc - 1, max(0, st * 1000 / FP))
            i0 = int(fi)
            i1 = min(nsrc - 1, i0 + 1)
            fr = fi - i0
            osp[k] = sp[i0] * (1 - fr) + sp[i1] * fr
            oap[k] = ap[i0] * (1 - fr) + ap[i1] * fr
            src_f = f0[i0] if f0[i0] > 0 else (f0[i1] if f0[i1] > 0 else 0)
            if src_f > 0:
                g = min(1.0, u / max(0.05, snd * 0.7))
                g = g * g * (3 - 2 * g)
                n = notes[0] + (notes[-1] - notes[0]) * g + shift
                n += -0.7 * max(0.0, 1 - u / 0.05)  # scoop into the note
                if snd > 0.38 and u > 0.22:
                    n += 0.22 * np.sin(2 * np.pi * 5.6 * (u - 0.22)) * min(1, (u - 0.22) / 0.15)
                of0[k] = midi_hz(n) * (src_f / med) ** keep
            gate[k] = min(1.0, (k - a) / 2.0, (b - k) / 3.0)
    # formant warp
    if abs(alpha - 1.0) > 1e-3:
        nb = osp.shape[1]
        src_bins = np.arange(nb) / alpha
        osp = np.array([np.interp(src_bins, np.arange(nb), row) for row in osp])
        oap = np.array([np.interp(src_bins, np.arange(nb), row) for row in oap])
    y = pw.synthesize(of0, np.ascontiguousarray(osp), np.ascontiguousarray(oap), FS, FP)
    g = np.repeat(gate, int(FS * FP / 1000))[:len(y)]
    g = np.pad(g, (0, max(0, len(y) - len(g))))
    y = y * g
    # even out word loudness (gentle)
    for tw in timings:
        a = int((tw['t0'] - t0_line) * FS)
        b = int((tw['t1'] - t0_line) * FS)
        seg = y[max(0, a):max(0, b)]
        if len(seg) > 100:
            r = np.sqrt((seg ** 2).mean()) + 1e-9
            y[max(0, a):max(0, b)] *= (0.08 / r) ** 0.6
    return y, t0_line, timings, score


def fx_chain(singer):
    if singer == 'dad':
        return Pedalboard([HighpassFilter(80), LowShelfFilter(cutoff_frequency_hz=180, gain_db=3.0), Compressor(threshold_db=-22, ratio=5, attack_ms=3, release_ms=80),
                           Distortion(drive_db=5), PeakFilter(cutoff_frequency_hz=2800, gain_db=4.0, q=0.9), LowpassFilter(11000),
                           Delay(delay_seconds=0.115, feedback=0.12, mix=0.12), Reverb(room_size=0.82, damping=0.45, wet_level=0.2, dry_level=0.85, width=1.0), Gain(-2)])
    return Pedalboard([HighpassFilter(140), Compressor(threshold_db=-22, ratio=4.5, attack_ms=3, release_ms=70), Distortion(drive_db=2.5),
                       PeakFilter(cutoff_frequency_hz=3200, gain_db=4.5, q=0.9), LowpassFilter(12000),
                       Delay(delay_seconds=0.115, feedback=0.12, mix=0.12), Reverb(room_size=0.8, damping=0.4, wet_level=0.2, dry_level=0.85, width=1.0), Gain(-2)])


def main():
    cfg = json.load(open(sys.argv[1]))
    beats = cfg['beats']
    dur = cfg['duration']
    fps = cfg.get('fps', 30)
    n = int(dur * OUT_SR)
    stems = {s: np.zeros((n, 2), np.float32) for s in ('dad', 'son')}
    dry = {s: np.zeros(n, np.float32) for s in ('dad', 'son')}
    cache = {}
    all_words = []
    report = []
    for li, line in enumerate(cfg['lines']):
        y, t0, timings, score = build_line(line, beats, cache)
        report.append((line['id'], line['singer'], round(score, 2)))
        y48 = resample_poly(y, 2, 1).astype(np.float32)
        layers = [(y48, 0.5, 1.0)]
        if line.get('double', True):
            l2 = dict(line, semitones=line.get('semitones', 0) + 0.09, formant=line.get('formant', 1.0) * 1.012)
            y2, _, _, _ = build_line(l2, beats, cache)
            y2 = resample_poly(y2, 2, 1).astype(np.float32)
            y2 = np.concatenate([np.zeros(int(0.014 * OUT_SR), np.float32), y2])
            layers += [(y2, 0.15, 0.55), (y2, 0.85, 0.0)]
            layers[0] = (y48, 0.5, 1.0)
        a = int(t0 * OUT_SR)
        gain = line.get('gain', 1.0)
        S = line['singer']
        for (yy, pan, g) in layers:
            if g <= 0:
                # mirror copy of double on the other side, slightly different delay
                yy = np.concatenate([np.zeros(int(0.009 * OUT_SR), np.float32), yy])
                g = 0.55
            b = min(n, a + len(yy))
            stems[S][a:b, 0] += yy[:b - a] * g * gain * np.sqrt(1 - pan)
            stems[S][a:b, 1] += yy[:b - a] * g * gain * np.sqrt(pan)
        b = min(n, a + len(y48))
        dry[S][a:b] += y48[:b - a]
        if line.get('subtitle', True):
            for tw in timings:
                tw['singer'] = line.get('sub_singer', S)
                tw['line'] = line['id']
            all_words += timings
    out = cfg['out_dir']
    for S in stems:
        st = stems[S]
        pk = np.abs(st).max()
        if pk < 1e-6:
            continue
        st = st / pk * 0.5
        proc = fx_chain(S)(st.T.copy(), OUT_SR).T
        proc = proc / (np.abs(proc).max() + 1e-9) * 0.89
        sf.write(f'{out}/vox_{S}.wav', proc, OUT_SR)
        # per-frame envelope for jaw animation (from the dry voice)
        hop = OUT_SR // fps
        d = dry[S]
        env = np.array([np.sqrt((d[i * hop:(i + 1) * hop] ** 2).mean()) for i in range(int(dur * fps))])
        env = env / (np.percentile(env[env > 0], 95) + 1e-9) if (env > 0).any() else env
        env = np.clip(env, 0, 1)
        # smooth attack/release
        sm = np.zeros_like(env)
        for i in range(len(env)):
            prev = sm[i - 1] if i else 0
            k = 0.7 if env[i] > prev else 0.35
            sm[i] = prev + (env[i] - prev) * k
        np.save(f'{out}/env_{S}.npy', sm.astype(np.float32))
    all_words.sort(key=lambda w: w['t0'])
    json.dump(all_words, open(f'{out}/words.json', 'w'), indent=1)
    print('alignment scores:', report)
    for take, (x, f0, sp, ap, heard) in cache.items():
        print(take.split('/')[-1], '->', ' '.join(h['w'] for h in heard))


if __name__ == '__main__':
    main()
