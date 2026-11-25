#!/usr/bin/env python3
"""The Braid Boys - 9:16 concert video compositor.

Builds a 1080x1920 video from a single still photo (dad + son on stage) plus a
timeline JSON (shots, beats, pyro hits, lyric word timings, vocal envelopes).

usage:
  compose.py DATA_DIR TIMELINE.json frames OUT_DIR t1,t2,...   # preview stills
  compose.py DATA_DIR TIMELINE.json video OUT.mp4 AUDIO.wav [workers]
"""
import os, sys, json, math, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

cv2.setNumThreads(1)
OW, OH = 1080, 1920
QW, QH = OW // 4, OH // 4
HERE = os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------- helpers
def sstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def fsstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def tile_noise(h, w, beta, seed):
    """Tileable fractal noise in [0,1] via spectral synthesis."""
    rng = np.random.default_rng(seed)
    F = np.fft.fft2(rng.standard_normal((h, w)))
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fx * fx + fy * fy)
    f[0, 0] = 1
    F = F / f ** beta
    F[0, 0] = 0
    n = np.real(np.fft.ifft2(F))
    lo, hi = np.percentile(n, [1, 99])
    return np.clip((n - lo) / (hi - lo), 0, 1).astype(np.float32)


def make_lut(stops):
    xs = np.linspace(0, 1, 256)
    lut = np.zeros((256, 3), np.float32)
    for c in range(3):
        lut[:, c] = np.interp(xs, [s[0] for s in stops], [s[1][c] for s in stops])
    return lut


FIRE_LUT = make_lut([(0.0, (0, 0, 0)), (0.2, (0.28, 0.02, 0.0)), (0.42, (0.9, 0.24, 0.02)),
                     (0.65, (1.0, 0.55, 0.08)), (0.85, (1.0, 0.82, 0.38)), (1.0, (1.0, 0.97, 0.86))])


def apply_lut(I, lut):
    idx = np.clip(I * 255, 0, 255).astype(np.uint8)
    return lut[idx]


def affine(cx, cy, vh, rot_deg, zoom=1.0, dx=0.0, dy=0.0):
    """Photo(1x) -> output pixels. Camera centred on (cx,cy) showing vh photo px vertically."""
    s = OH / vh * zoom
    a = math.radians(rot_deg)
    c, sn = math.cos(a) * s, math.sin(a) * s
    return np.array([[c, -sn, OW / 2 + dx - (c * cx - sn * cy)],
                     [sn, c, OH / 2 + dy - (sn * cx + c * cy)]], np.float32)


def compose_aff(A, B):
    A3 = np.vstack([A, [0, 0, 1]])
    B3 = np.vstack([B, [0, 0, 1]])
    return (A3 @ B3)[:2].astype(np.float32)


def apply_pts(M, pts):
    pts = np.asarray(pts, np.float32).reshape(-1, 2)
    return pts @ M[:, :2].T + M[:, 2]


def pulse_train(times, t, decay):
    """Exponential decay from the most recent event time <= t."""
    if len(times) == 0:
        return 0.0
    i = np.searchsorted(times, t, side='right') - 1
    if i < 0:
        return 0.0
    return math.exp(-(t - times[i]) / decay)


def over(dst, rgb, a):
    """alpha-composite float images in place: dst = dst*(1-a) + rgb*a"""
    dst *= (1 - a)
    dst += rgb * a
    return dst


# ----------------------------------------------------------------- text
class TextFx:
    def __init__(self, font_dir):
        self.font = ImageFont.truetype(os.path.join(font_dir, 'Anton-Regular.ttf'), 112)
        self.logo_font = ImageFont.truetype(os.path.join(font_dir, 'MetalMania-Regular.ttf'), 150)
        self.small_font = ImageFont.truetype(os.path.join(font_dir, 'Anton-Regular.ttf'), 54)
        self.cache = {}

    def word(self, text, fill, stroke=(0, 0, 0)):
        key = (text, fill)
        if key in self.cache:
            return self.cache[key]
        f = self.font
        bb = f.getbbox(text, stroke_width=10)
        w, h = bb[2] - bb[0] + 40, bb[3] - bb[1] + 50
        im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        # soft drop shadow
        sh = Image.new('L', (w, h), 0)
        ImageDraw.Draw(sh).text((20 - bb[0], 32 - bb[1]), text, font=f, fill=255, stroke_width=12, stroke_fill=255)
        sh = sh.filter(ImageFilter.GaussianBlur(7))
        shadow = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        shadow.putalpha(sh.point(lambda v: int(v * 0.65)))
        im = Image.alpha_composite(im, shadow)
        d = ImageDraw.Draw(im)
        d.text((20 - bb[0], 20 - bb[1]), text, font=f, fill=fill + (255,), stroke_width=10, stroke_fill=stroke + (255,))
        a = np.asarray(im).astype(np.float32) / 255.0
        self.cache[key] = a
        return a

    def logo(self, text='THE BRAID BOYS', size=150, sub=None):
        key = ('logo', text, size, sub)
        if key in self.cache:
            return self.cache[key]
        f = ImageFont.truetype(self.logo_font.path, size)
        bb = f.getbbox(text, stroke_width=8)
        w, h = bb[2] - bb[0] + 120, bb[3] - bb[1] + 120
        mask = Image.new('L', (w, h), 0)
        ImageDraw.Draw(mask).text((60 - bb[0], 60 - bb[1]), text, font=f, fill=255)
        strokem = Image.new('L', (w, h), 0)
        ImageDraw.Draw(strokem).text((60 - bb[0], 60 - bb[1]), text, font=f, fill=255, stroke_width=8, stroke_fill=255)
        m = np.asarray(mask).astype(np.float32) / 255
        sm = np.asarray(strokem).astype(np.float32) / 255
        yy = np.linspace(0, 1, h)[:, None] * np.ones((1, w))
        n = tile_noise(h, w, 1.3, 7)
        heat = np.clip(1.05 - yy * 0.9 + (n - 0.5) * 0.35, 0, 1)
        fill = apply_lut(0.45 + heat * 0.55, FIRE_LUT)
        glow = cv2.GaussianBlur(sm, (0, 0), 18)
        rgb = np.zeros((h, w, 3), np.float32)
        rgb += glow[..., None] * np.array([1.0, 0.35, 0.05]) * 0.9
        a = np.clip(glow * 0.9, 0, 1)
        # dark stroke then fire fill
        rgb = rgb * (1 - sm[..., None]) + np.array([0.06, 0.02, 0.0]) * sm[..., None]
        a = np.maximum(a, sm)
        rgb = rgb * (1 - m[..., None]) + fill * m[..., None]
        out = np.dstack([rgb, a]).astype(np.float32)
        self.cache[key] = out
        return out

    def small(self, text, fill=(255, 255, 255)):
        key = ('small', text, fill)
        if key in self.cache:
            return self.cache[key]
        f = self.small_font
        bb = f.getbbox(text, stroke_width=6)
        w, h = bb[2] - bb[0] + 30, bb[3] - bb[1] + 30
        im = Image.new('RGBA', (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((15 - bb[0], 15 - bb[1]), text, font=f, fill=fill + (255,), stroke_width=6, stroke_fill=(0, 0, 0, 255))
        a = np.asarray(im).astype(np.float32) / 255.0
        self.cache[key] = a
        return a


def paste_rgba(dst, src, cx, cy, scale=1.0, alpha=1.0):
    """Composite float RGBA `src` centred at (cx,cy) onto float RGB dst."""
    if alpha <= 0.003:
        return
    if abs(scale - 1.0) > 1e-3:
        h, w = src.shape[:2]
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        src = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_LINEAR if scale > 1 else cv2.INTER_AREA)
    h, w = src.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(dst.shape[1], x0 + w), min(dst.shape[0], y0 + h)
    if xa >= xb or ya >= yb:
        return
    s = src[ya - y0:yb - y0, xa - x0:xb - x0]
    a = s[..., 3:4] * alpha
    d = dst[ya:yb, xa:xb]
    d *= (1 - a)
    d += s[..., :3] * a


# ----------------------------------------------------------------- crowd sprites
def draw_person(d, x, y, r, rng, arms, phone, lit, sil=(10, 7, 7)):
    """Silhouette: head at (x,y) radius r; shoulders below. Returns phone screen pos list."""
    screens = []
    # shoulders / torso
    d.ellipse([x - r * 2.1, y + r * 0.9, x + r * 2.1, y + r * 5.5], fill=sil)
    d.ellipse([x - r, y - r * 1.1, x + r, y + r * 1.05], fill=sil)
    for side in arms:
        sx = x + side * r * 1.6
        sy = y + r * 1.6
        hx = x + side * r * rng.uniform(1.2, 2.8)
        hy = y - r * rng.uniform(2.2, 3.6)
        wdt = max(2, int(r * 0.75))
        d.line([sx, sy, hx, hy], fill=sil, width=wdt)
        if phone and side == arms[0]:
            pw, ph = r * 0.75, r * 1.25
            d.rectangle([hx - pw / 2, hy - ph, hx + pw / 2, hy], fill=sil)
            screens.append((hx, hy - ph / 2, pw * 0.8, ph * 0.85))
        else:
            hr = r * 0.62
            d.ellipse([hx - hr, hy - hr * 1.2, hx + hr, hy + hr * 0.9], fill=sil)
            if rng.random() < 0.35:  # horns up
                d.line([hx - hr * 0.6, hy - hr, hx - hr * 0.8, hy - hr * 2.6], fill=sil, width=max(1, int(hr * 0.45)))
                d.line([hx + hr * 0.6, hy - hr, hx + hr * 0.8, hy - hr * 2.6], fill=sil, width=max(1, int(hr * 0.45)))
    return screens


def crowd_strip(width, r, n, rng, rim_col, height_mul=6.5, arm_p=0.5, phone_p=0.12, ss=2):
    """Pre-render a horizontal strip of silhouettes as float RGBA (supersampled)."""
    W, H = width * ss, int(r * height_mul * 2.2) * ss
    base_y = int(r * height_mul * 1.25) * ss
    im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    screens = []
    xs = np.sort(rng.uniform(0, W, n))
    for x in xs:
        rr = r * ss * rng.uniform(0.85, 1.15)
        y = base_y + rng.uniform(-0.5, 0.5) * rr
        a = []
        if rng.random() < arm_p:
            a = [rng.choice([-1, 1])] if rng.random() < 0.6 else [-1, 1]
        screens += draw_person(d, x, y, rr, rng, a, rng.random() < phone_p and len(a) > 0, rim_col)
    arr = np.asarray(im).astype(np.float32) / 255.0
    alpha = arr[..., 3]
    # rim light on upper edges
    shift = max(1, int(r * ss * 0.18))
    up = np.zeros_like(alpha)
    up[shift:] = alpha[:-shift]
    rim = np.clip(alpha - up, 0, 1)
    rim = cv2.GaussianBlur(rim, (0, 0), max(0.6, r * ss * 0.06))
    rgb = arr[..., :3] + rim[..., None] * np.array(rim_col, np.float32)[None, None] * 1.4
    glow = np.zeros_like(alpha)
    for (sx, sy, pw, ph) in screens:
        cv2.rectangle(rgb, (int(sx - pw / 2), int(sy - ph / 2)), (int(sx + pw / 2), int(sy + ph / 2)), (0.85, 0.92, 1.0), -1)
        cv2.circle(glow, (int(sx), int(sy)), int(ph * 1.6), 1.0, -1)
    if screens:
        glow = cv2.GaussianBlur(glow, (0, 0), r * ss * 0.8) * 0.6
        rgb += glow[..., None] * np.array([0.7, 0.85, 1.0])
        alpha = np.maximum(alpha, glow * 0.5)
    out = np.dstack([np.clip(rgb, 0, 2), alpha])
    out = cv2.resize(out, (width, H // ss), interpolation=cv2.INTER_AREA)
    return out, base_y // ss


# ----------------------------------------------------------------- renderer
class Renderer:
    def __init__(self, D, tl, font_dir):
        self.D, self.tl = D, tl
        self.fps = tl['fps']
        self.meta = json.load(open(f'{D}/meta.json'))
        self.W0, self.H0 = self.meta['W0'], self.meta['H0']
        self.photo2 = np.load(f'{D}/photo2.npy')
        self.bg2 = np.load(f'{D}/bg2.npy')
        self.son_a2 = np.load(f'{D}/son_a2.npy')
        self.dad_a2 = np.load(f'{D}/dad_a2.npy')
        self.flame1 = np.load(f'{D}/flame_soft.npy').astype(np.float32) / 255.0
        self.text = TextFx(font_dir)
        self.beats = np.array(tl['beats'], np.float64)
        self.downbeats = np.array(tl.get('downbeats', []), np.float64)
        self.hits = np.array([h['t'] for h in tl['hits']], np.float64)
        self.hit_amp = np.array([h.get('a', 1.0) for h in tl['hits']], np.float64)
        self.strobes = np.array(tl.get('strobes', []), np.float64)
        self.env = {k: np.array(v, np.float32) for k, v in tl['env'].items()}
        rng = np.random.default_rng(3)
        self.n_fire = [tile_noise(256, 128, 1.15, s) for s in (11, 12, 13)]
        self.n_disp = [tile_noise(256, 256, 1.6, s) for s in (21, 22)]
        self.n_haze = tile_noise(256, 512, 1.9, 31)
        self.grain = [rng.normal(0, 1, (OH // 2, OW // 2)).astype(np.float32) for _ in range(6)]
        yy, xx = np.mgrid[0:OH, 0:OW].astype(np.float32)
        self.gx, self.gy = xx, yy
        r2 = ((xx - OW / 2) / (OW * 0.75)) ** 2 + ((yy - OH / 2) / (OH * 0.72)) ** 2
        self.vignette = np.clip(1 - 0.42 * r2, 0.35, 1)[..., None].astype(np.float32)
        qy, qx = np.mgrid[0:QH, 0:QW].astype(np.float32)
        self.qx, self.qy = qx, qy
        # pyro jets: bases (photo coords) taken from flame columns
        self.jets = tl.get('jets', [])
        # stage lamps
        self.lamps = np.array([[l['x'], l['y']] for l in self.meta['lamps']], np.float32)
        # particles (sparks) in photo space
        self._init_sparks()
        # foreground crowd strips (screen space)
        self.fg_strips = []
        for i, (r, n, col) in enumerate([(30, 16, (1.0, 0.6, 0.25)), (46, 10, (1.0, 0.55, 0.2))]):
            strip, base = crowd_strip(OW + 200, r, n, np.random.default_rng(100 + i), col, arm_p=0.7, phone_p=0.28)
            strip = cv2.GaussianBlur(strip, (0, 0), 5.0 + 4.0 * i)  # strong depth of field: out-of-focus front rows
            strip[..., 3] *= 0.92
            self.fg_strips.append((strip, base, i))
        self._init_arena()

    # ---------------- sparks
    def _init_sparks(self):
        rng = np.random.default_rng(5)
        N = 2600
        dur = self.tl['duration']
        jets = self.jets or [{'x': 320, 'y': 860}]
        src = rng.integers(0, len(jets), N)
        self.sp_t0 = rng.uniform(-2, dur, N)
        # bias spawns around pyro hits
        if len(self.hits):
            k = N // 2
            hh = rng.integers(0, len(self.hits), k)
            self.sp_t0[:k] = self.hits[hh] + rng.exponential(0.25, k)
        self.sp_x0 = np.array([jets[s]['x'] for s in src]) + rng.normal(0, 25, N)
        self.sp_y0 = np.array([jets[s]['y'] - jets[s].get('h', 700) * rng.uniform(0.3, 1.0) for s in src])
        self.sp_vx = rng.normal(0, 60, N)
        self.sp_vy = -rng.uniform(80, 320, N)
        self.sp_life = rng.uniform(0.8, 2.6, N)
        self.sp_b = rng.uniform(0.4, 1.0, N)
        self.sp_ph = rng.uniform(0, 6.28, N)

    # ---------------- arena (reverse angle crowd shot), rendered at half res
    def _init_arena(self):
        AW, AH = OW // 2, OH // 2
        self.AW, self.AH = AW, AH
        rng = np.random.default_rng(9)
        y = np.linspace(0, 1, AH)[:, None]
        base = np.zeros((AH, AW, 3), np.float32)
        top, mid, bot = np.array([0.02, 0.025, 0.06]), np.array([0.09, 0.05, 0.08]), np.array([0.16, 0.07, 0.03])
        for i in range(AH):
            t = i / AH
            base[i] = lerp(top, mid, t / 0.5) if t < 0.5 else lerp(mid, bot, (t - 0.5) / 0.5)
        hz = cv2.resize(tile_noise(128, 128, 2.0, 41), (AW, AH))
        base *= (0.75 + 0.5 * hz[..., None])
        self.ar_base = base
        # stands: tiers of phone lights in the upper bowl
        pts = []
        for tier in range(11):
            y0 = AH * (0.10 + tier * 0.033)
            n = 160 + tier * 25
            xs = rng.uniform(-20, AW + 20, n)
            curve = -((xs - AW / 2) / (AW / 2)) ** 2 * AH * 0.05
            ys = y0 + curve + rng.normal(0, 2.5, n)
            pts.append(np.stack([xs, ys, np.full(n, 0.5 + tier * 0.05)], 1))
        P = np.concatenate(pts)
        self.ar_pts = P
        self.ar_ph = rng.uniform(0, 6.28, len(P))
        self.ar_fr = rng.uniform(0.3, 2.5, len(P))
        self.ar_on = rng.random(len(P)) < 0.55
        self.ar_col = np.where(rng.random(len(P))[:, None] < 0.8, np.array([[0.8, 0.88, 1.0]]), np.array([[1.0, 0.75, 0.4]])).astype(np.float32)
        # floor crowd rows (back to front)
        self.ar_rows = []
        nrows = 13
        for i in range(nrows):
            s = (i + 1) / nrows
            r = 2.2 + 30 * s ** 2.2
            n = int(AW / (r * 2.0) * 1.15) + 2
            col = lerp(np.array([0.9, 0.45, 0.6]), np.array([1.0, 0.55, 0.2]), s)
            strip, by = crowd_strip(AW + 80, r, n, np.random.default_rng(200 + i), tuple(col), arm_p=0.35 + 0.3 * s, phone_p=0.18)
            ybase = AH * (0.44 + 0.62 * s ** 1.35)
            fog = (1 - s) * 0.55
            self.ar_rows.append((strip, by, ybase, s, fog))
        ay, ax = np.mgrid[0:AH, 0:AW].astype(np.float32)
        ox, oy = AW / 2, AH * 1.25
        self.ar_ang = np.arctan2(ax - ox, oy - ay)
        self.ar_dist = np.hypot(ax - ox, ay - oy)

    def arena(self, t, cam):
        AW, AH = self.AW, self.AH
        img = self.ar_base.copy()
        beat = pulse_train(self.beats, t, 0.16)
        hit = self.hit(t)
        # phone lights
        P = self.ar_pts
        sway = np.sin(t * 1.7 + P[:, 1] * 0.05) * 3
        b = (0.55 + 0.45 * np.sin(t * self.ar_fr + self.ar_ph)) * self.ar_on * P[:, 2]
        lights = np.zeros((AH, AW, 3), np.float32)
        xi = np.clip((P[:, 0] + sway).astype(int), 0, AW - 1)
        yi = np.clip(P[:, 1].astype(int), 0, AH - 1)
        np.add.at(lights, (yi, xi), self.ar_col * b[:, None] * 2.2)
        lights = cv2.GaussianBlur(lights, (0, 0), 0.9) + cv2.GaussianBlur(lights, (0, 0), 3.5) * 0.6
        img += lights
        # beams fanning out from the stage behind camera
        beams = np.zeros((AH, AW), np.float32)
        beams_w = np.zeros((AH, AW), np.float32)
        for k in range(6):
            th = math.sin(t * (0.55 + 0.13 * k) + k * 1.9) * 0.75
            w = 0.045 + 0.01 * (k % 3)
            prof = np.exp(-0.5 * ((self.ar_ang - th) / w) ** 2) * np.clip(1 - self.ar_dist / (AH * 1.6), 0, 1)
            (beams if k % 2 else beams_w)[:] += prof
        hzn = cv2.resize(np.roll(self.n_haze, int(t * 20), axis=1), (AW, AH))
        haze = 0.55 + 0.6 * hzn
        img += (beams[..., None] * np.array([1.0, 0.55, 0.2]) + beams_w[..., None] * np.array([0.75, 0.8, 1.0])) * haze[..., None] * (0.32 + 0.25 * beat)
        # crowd rows, back to front, jumping on the beat
        bp = 60.0 / max(1e-3, self.tl.get('bpm', 130))
        for i, (strip, by, ybase, s, fog) in enumerate(self.ar_rows):
            phase = (t / bp + (i % 3) * 0.17) % 1.0
            jump = math.sin(math.pi * phase) ** 2 * (2 + 14 * s) * (0.6 + 0.4 * beat)
            dx = math.sin(t * 0.9 + i) * 6 * s
            sh, sw = strip.shape[:2]
            y0 = int(ybase - by - jump)
            x0 = int(-40 + dx)
            ya, yb = max(0, y0), min(AH, y0 + sh)
            xa, xb = max(0, x0), min(AW, x0 + sw)
            if ya >= yb:
                continue
            S = strip[ya - y0:yb - y0, xa - x0:xb - x0]
            a = S[..., 3:4]
            rgb = S[..., :3] * (0.7 + 0.6 * hit + 0.3 * beat)
            rgb = lerp(rgb, np.array([0.22, 0.12, 0.16], np.float32), fog)
            d = img[ya:yb, xa:xb]
            d *= (1 - a)
            d += rgb * a
        # pyro glow from the stage (bottom) on hits
        g = np.clip(np.linspace(-0.6, 1.0, AH), 0, 1)[:, None, None] ** 1.5
        img += g * np.array([1.0, 0.45, 0.1]) * (0.25 * hit + 0.06 * beat)
        img = np.clip(img, 0, 4)
        # camera: push + shake, upscale to output
        z = cam.get('zoom', 1.0)
        M = np.array([[2 * z, 0, OW / 2 - AW * z + cam.get('dx', 0)], [0, 2 * z, OH / 2 - AH * z + cam.get('dy', 0)]], np.float32)
        a = math.radians(cam.get('rot', 0))
        R = cv2.getRotationMatrix2D((OW / 2, OH / 2), math.degrees(a), 1.0).astype(np.float32)
        M = compose_aff(R, M)
        out = cv2.warpAffine(img, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return out

    # ---------------- generated crowd photos (Ken Burns + live lighting)
    def _img(self, name):
        if not hasattr(self, '_imgs'):
            self._imgs = {}
        if name not in self._imgs:
            im = cv2.cvtColor(cv2.imread(f'{self.D}/crowd/{name}.png'), cv2.COLOR_BGR2RGB)
            blur = cv2.GaussianBlur(im, (0, 0), 1.4)
            im = cv2.addWeighted(im, 1.35, blur, -0.35, 0)
            self._imgs[name] = im
        return self._imgs[name]

    def image_shot(self, t, s, cam):
        img = self._img(s['img'])
        H, W = img.shape[:2]
        z = max(OW / W, OH / H) * cam['zoom']
        px, py = cam.get('px', 0.0), cam.get('py', 0.0)
        M = np.array([[z, 0, OW / 2 - z * (W / 2 + px * W) + cam.get('dx', 0)],
                      [0, z, OH / 2 - z * (H / 2 + py * H) + cam.get('dy', 0)]], np.float32)
        R = cv2.getRotationMatrix2D((OW / 2, OH / 2), cam.get('rot', 0), 1.0).astype(np.float32)
        M = compose_aff(R, M)
        out = cv2.warpAffine(img, M, (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).astype(np.float32) / 255
        hit = self.hit(t)
        beat = pulse_train(self.beats, t, 0.15)
        out *= (1 + hit * np.array([0.42, 0.2, 0.04], np.float32) + 0.1 * beat)
        # sweeping light from the stage rig across the crowd
        q = np.zeros((QH, QW), np.float32)
        for k in range(3):
            cx = QW * (0.5 + 0.55 * math.sin(t * (0.9 + 0.3 * k) + k * 2.0))
            q += np.exp(-0.5 * ((self.qx - cx - (self.qy - QH) * 0.25 * (k - 1)) / (QW * 0.07)) ** 2)
        q = cv2.resize(q, (OW, OH))
        out += q[..., None] * out * np.array([0.35, 0.3, 0.25], np.float32) * (0.5 + beat)
        return out

    # ---------------- timeline queries
    def hit(self, t, decay=0.45):
        if len(self.hits) == 0:
            return 0.0
        i = np.searchsorted(self.hits, t, side='right') - 1
        if i < 0:
            return 0.0
        dt = t - self.hits[i]
        return float(self.hit_amp[i] * math.exp(-dt / decay))

    def envelope(self, name, t):
        e = self.env.get(name)
        if e is None or len(e) == 0:
            return 0.0
        i = int(t * self.fps)
        return float(e[min(max(i, 0), len(e) - 1)])

    def shot_at(self, t):
        for s in self.tl['shots']:
            if s['t0'] <= t < s['t1']:
                return s
        return self.tl['shots'][-1]

    def camera(self, s, t):
        u = (t - s['t0']) / max(1e-3, s['t1'] - s['t0'])
        e = s.get('ease', 'io')
        u = u * u * (3 - 2 * u) if e == 'io' else (1 - (1 - u) ** 3 if e == 'out' else u)
        A, B = s['a'], s.get('b', s['a'])
        cam = {k: lerp(A.get(k, 0), B.get(k, A.get(k, 0)), u) for k in set(A) | set(B)}
        cam.setdefault('rot', 0.0)
        cam['zoom'] = cam.get('zoom', 1.0)
        # beat punch zoom
        punch = s.get('punch', 0.03) * pulse_train(self.beats, t, 0.12) + 0.06 * self.hit(t, 0.2) * s.get('hitpunch', 1.0)
        # entrance zoom (explosive transitions)
        tin = t - s['t0']
        ent = s.get('enter', 0.0)
        cam['zoom'] *= 1 + punch + ent * math.exp(-tin / 0.09)
        # handheld shake (sum of sines), stronger on hits
        sh = s.get('shake', 6.0) * (1 + 1.6 * self.hit(t, 0.3))
        cam['dx'] = cam.get('dx', 0) + sh * (math.sin(t * 7.3 + 1.1) * 0.6 + math.sin(t * 13.1 + 0.3) * 0.4)
        cam['dy'] = cam.get('dy', 0) + sh * (math.sin(t * 6.1 + 2.3) * 0.6 + math.sin(t * 15.7 + 1.7) * 0.4)
        cam['rot'] += sh * 0.02 * math.sin(t * 4.3 + 0.5)
        # whip pan in/out
        for key, sign in (('whip_in', -1), ('whip_out', 1)):
            if s.get(key):
                d = (t - s['t0']) if key == 'whip_in' else (s['t1'] - t)
                if d < 0.12:
                    k = (1 - d / 0.12) ** 2
                    cam['dx'] += sign * s.get('whip_dir', 1) * k * OW * 0.6
                    cam['whip'] = max(cam.get('whip', 0), k)
        return cam

    def clamp_cam(self, cam):
        vh = cam['vh'] / cam['zoom']
        a = math.radians(cam['rot'])
        hw, hh = vh * 9 / 16 / 2, vh / 2
        ex = abs(math.cos(a)) * hw + abs(math.sin(a)) * hh
        ey = abs(math.sin(a)) * hw + abs(math.cos(a)) * hh
        f = max(1.0, ex / (self.W0 / 2), ey / (self.H0 / 2))
        if f > 1.0:
            cam['zoom'] *= f
            ex, ey = ex / f, ey / f
        s = OH / cam['vh'] * cam['zoom']
        # convert pixel shake into photo units and clamp centre
        cx = cam['cx'] - cam.get('dx', 0) / s
        cy = cam['cy'] - cam.get('dy', 0) / s
        cx = min(max(cx, ex), self.W0 - ex)
        cy = min(max(cy, ey), self.H0 - ey)
        return affine(cx, cy, cam['vh'], cam['rot'], cam['zoom'])

    # ---------------- stage render (photo based)
    def stage(self, t, cam):
        M = self.clamp_cam(cam)
        s = abs(M[0, 0] + 1j * M[1, 0])
        H2 = np.array([[0.5, 0, 0], [0, 0.5, 0]], np.float32)
        M2 = compose_aff(M, H2)
        beat = pulse_train(self.beats, t, 0.15)
        hit = self.hit(t)
        bg = cv2.warpAffine(self.bg2, M2, (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).astype(np.float32) / 255.0
        fm = cv2.warpAffine(self.flame1, M, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # living flames: upward-scrolling turbulence inside the flame regions
        sc = s / 1.05
        n1 = np.roll(self.n_disp[0], int(-t * 90) % 256, axis=0)
        n2 = np.roll(self.n_disp[1], int(-t * 140) % 256, axis=0)
        dxs = cv2.resize(n1, (OW // 8, OH // 8))
        dys = cv2.resize(n2, (OW // 8, OH // 8))
        dx = cv2.resize(dxs, (OW, OH)) - 0.5
        dy = cv2.resize(dys, (OW, OH)) - 0.5
        amp = 22 * sc * (1 + 0.8 * hit)
        mx = self.gx + dx * amp * fm
        my = self.gy + (dy * amp * 1.4 + 6 * sc) * fm
        bg = cv2.remap(bg, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        flick = 1 + (0.18 * math.sin(t * 23) + 0.1 * math.sin(t * 37 + 1) + 0.55 * hit + 0.15 * beat)
        bg *= (1 + fm[..., None] * (flick - 1))
        add = np.zeros((OH, OW, 3), np.float32)
        # pyro jets (additive, behind performers)
        for j, jet in enumerate(self.jets):
            hj = self._jet_height(t, j)
            if hj <= 0.02:
                continue
            tex = self._fire_tex(t, j, hj)
            th, tw = tex.shape[:2]
            Wj, Hj = jet.get('w', 150), jet.get('h', 700)
            # jet texture -> photo coords
            J = np.array([[Wj / tw, 0, jet['x'] - Wj / 2], [0, Hj / th, jet['y'] - Hj]], np.float32)
            MJ = compose_aff(M, J)
            add += cv2.warpAffine(tex, MJ, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        # light beams + lamp glows (quarter res)
        q = np.zeros((QH, QW, 3), np.float32)
        L = apply_pts(M, self.lamps) / 4.0
        chase = self.tl.get('bpm', 130) / 60.0
        energy = self.envelope('energy', t)
        for i, (lx, ly) in enumerate(L):
            on = 0.45 + 0.55 * (0.5 + 0.5 * math.sin(t * chase * math.pi + i * 0.9))
            on = on * (0.6 + 0.8 * beat) * (0.5 + energy)
            g = s / 4.0
            rad = max(1.0, 3.0 * g)
            x0, x1 = int(max(0, lx - rad * 4)), int(min(QW, lx + rad * 4 + 1))
            y0, y1 = int(max(0, ly - rad * 4)), int(min(QH, ly + rad * 4 + 1))
            if x0 >= x1 or y0 >= y1:
                continue
            gx = self.qx[y0:y1, x0:x1] - lx
            gy = self.qy[y0:y1, x0:x1] - ly
            q[y0:y1, x0:x1] += np.exp(-(gx * gx + gy * gy) / (2 * rad * rad))[..., None] * np.array([1.0, 0.9, 0.75]) * on * 0.9
        # sweeping beams from the truss
        src_pts = apply_pts(M, np.array(self.tl.get('beam_src', [[250, 400], [450, 400], [650, 400], [880, 420]]), np.float32)) / 4.0
        for k, (bx, by) in enumerate(src_pts):
            th = math.sin(t * (0.8 + 0.21 * k) + k * 2.1) * 0.7 + (0.25 if k % 2 else -0.25)
            ang = np.arctan2(self.qx - bx, by - self.qy)
            dist = np.hypot(self.qx - bx, self.qy - by)
            w = 0.05 + 0.02 * (k % 2)
            prof = np.exp(-0.5 * ((ang - th) / w) ** 2) * np.clip(dist / 20, 0, 1) * np.exp(-dist / (QH * 0.7))
            col = np.array([0.75, 0.85, 1.0]) if k % 2 else np.array([1.0, 0.7, 0.35])
            q += prof[..., None] * col * (0.18 + 0.28 * beat) * (0.4 + energy)
        q = cv2.GaussianBlur(q, (0, 0), 1.2)
        add += cv2.resize(q, (OW, OH), interpolation=cv2.INTER_LINEAR)
        # haze drifting across the stage, lit warm
        hz = cv2.resize(np.roll(self.n_haze, int(t * 30), axis=1)[:, :256], (OW, OH))
        add += (hz[..., None] * np.array([1.0, 0.55, 0.25]) * (0.04 + 0.08 * hit))
        # people mask in output space (for occlusion)
        son_cam = compose_aff(M2, self._bounce('son', t))
        dad_cam = compose_aff(M2, self._bounce('dad', t))
        son_a = cv2.warpAffine(self.son_a2, son_cam, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)[..., None] / 255
        dad_a = cv2.warpAffine(self.dad_a2, dad_cam, (OW, OH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(np.float32)[..., None] / 255
        bg += add  # background fx
        frame = bg
        warm = 1 + hit * np.array([0.38, 0.17, 0.02], np.float32) + beat * 0.05
        for name, Mc, a in (('son', son_cam, son_a), ('dad', dad_cam, dad_a)):
            if a.max() < 0.01:
                continue
            rgb = cv2.warpAffine(self.photo2, Mc, (OW, OH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT).astype(np.float32) / 255
            rgb = self._jaw(name, rgb, t, compose_aff(M, self._bounce1(name, t)), s)
            rgb = rgb * warm
            over(frame, rgb, a)
        self._eyes(frame, t, M, s)
        self._sparks(frame, t, M, s)
        self._floor_fog(frame, t, M)
        return frame

    def _bounce1(self, who, t):
        """bounce affine in photo(1x) space"""
        bp = 60.0 / max(1e-3, self.tl.get('bpm', 130))
        ph = (t / bp) % 1.0
        sing = self.envelope(who, t)
        if who == 'son':
            fx, fy = 270, 1830
            amp = 4.0 * (0.4 + 0.6 * self.envelope('energy', t))
            ty = -amp * math.sin(math.pi * ph) ** 2
            rot = 0.35 * math.sin(2 * math.pi * t / (2 * bp))
            sc = 1 + 0.004 * sing
        else:
            fx, fy = 820, 1830
            amp = 2.5 * (0.4 + 0.6 * self.envelope('energy', t))
            ty = -amp * math.sin(math.pi * (ph + 0.15)) ** 2
            rot = -0.3 * math.sin(2 * math.pi * t / (2 * bp) + 0.7)
            sc = 1 + 0.003 * sing
        R = cv2.getRotationMatrix2D((fx, fy), rot, sc).astype(np.float32)
        R[1, 2] += ty
        return R

    def _bounce(self, who, t):
        R = self._bounce1(who, t)
        # in 2x source space: S2^-1 * R * S2 where S2 = 0.5 scale
        A = np.array([[2, 0, 0], [0, 2, 0]], np.float32)
        H2 = np.array([[0.5, 0, 0], [0, 0.5, 0]], np.float32)
        return compose_aff(A, compose_aff(R, H2))

    def _jaw(self, who, rgb, t, Mp, s):
        """Drop the jaw with the vocal envelope by a local warp around the chin."""
        e = self.envelope(who, t)
        if e < 0.03:
            return rgb
        J = self.tl['faces'][who]
        mx, my = apply_pts(Mp, [J['mouth']])[0]
        cxp, cyp = apply_pts(Mp, [J['chin']])[0]
        rx, ry = J['rx'] * s, J['ry'] * s
        amp = J['amp'] * s * e
        x0, x1 = int(max(0, cxp - rx * 2.5)), int(min(OW, cxp + rx * 2.5))
        y0, y1 = int(max(0, my - ry * 1.2)), int(min(OH, cyp + ry * 2.5))
        if x1 - x0 < 4 or y1 - y0 < 4:
            return rgb
        gx = self.gx[y0:y1, x0:x1]
        gy = self.gy[y0:y1, x0:x1]
        w = np.exp(-(((gx - cxp) / rx) ** 2 + ((gy - cyp) / ry) ** 2))
        w *= sstep(my - ry * 0.25, my + ry * 0.35, gy)  # keep upper lip/nose fixed
        if 'block_x' in J:  # protect hand/mic region (photo coords -> output)
            bx = apply_pts(Mp, [[J['block_x'], J['block_y']]])[0]
            w *= 1 - sstep(bx[0] - 6 * s, bx[0] + 6 * s, gx) * (1 - sstep(bx[1] - 4 * s, bx[1] + 8 * s, gy))
        mxp = gx.astype(np.float32)
        myp = (gy - amp * w).astype(np.float32)
        patch = cv2.remap(rgb, mxp, myp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # darken the opened mouth cavity slightly
        cav = np.exp(-(((gx - mx) / (rx * 0.6)) ** 2 + ((gy - my - amp * 0.4) / (ry * 0.35 + 1)) ** 2)) * min(1.0, e * 1.2)
        patch *= (1 - 0.25 * cav[..., None])
        rgb[y0:y1, x0:x1] = patch
        return rgb

    def _eyes(self, frame, t, M, s):
        g = 0.0
        for gl in self.tl.get('eye_glints', []):
            if gl['t0'] <= t < gl['t1']:
                u = (t - gl['t0']) / (gl['t1'] - gl['t0'])
                g = max(g, math.sin(math.pi * u) ** 0.6)
                who = gl['who']
        if g <= 0.01:
            return
        for (ex, ey) in self.tl['faces'][who]['eyes']:
            x, y = apply_pts(M, [[ex, ey]])[0]
            r = 7 * s
            x0, x1 = int(max(0, x - r * 14)), int(min(OW, x + r * 14))
            y0, y1 = int(max(0, y - r * 3)), int(min(OH, y + r * 3))
            if x0 >= x1 or y0 >= y1:
                continue
            gx = self.gx[y0:y1, x0:x1] - x
            gy = self.gy[y0:y1, x0:x1] - y
            core = np.exp(-(gx * gx + gy * gy) / (2 * (r * 0.55) ** 2))
            streak = np.exp(-(gy * gy) / (2 * (r * 0.15) ** 2)) * np.exp(-np.abs(gx) / (r * 2.6))
            frame[y0:y1, x0:x1] += (core[..., None] * 0.8 + streak[..., None] * 0.42) * np.array([0.45, 0.8, 1.0]) * g

    def _jet_height(self, t, j):
        """0..1 height of extra pyro jet j: shoots up on hits, then falls."""
        jet = self.jets[j]
        best = 0.0
        for h, a in zip(self.hits, self.hit_amp):
            if jet.get('only') and h not in jet['only']:
                pass
            dt = t - h
            if -0.02 < dt < 1.4:
                up = fsstep(-0.02, 0.12, dt)
                down = 1 - fsstep(0.35, 1.3, dt)
                best = max(best, a * up * down)
        return best

    def _fire_tex(self, t, j, hj):
        th, tw = 256, 96
        N1 = self.n_fire[j % 3]
        N2 = self.n_fire[(j + 1) % 3]
        off1 = int(t * 520 + j * 37) % N1.shape[0]
        off2 = int(t * 300 + j * 91) % N2.shape[0]
        n1 = np.roll(N1, off1, axis=0)[:th, :tw]
        n2 = np.roll(N2, off2, axis=0)[:th, :tw]
        v = 1 - np.linspace(0, 1, th)[:, None]  # 0 at base(bottom) .. 1 top
        u = np.linspace(-1, 1, tw)[None, :]
        wv = 0.22 + 0.7 * v ** 0.7
        uu = u + (n2 - 0.5) * 0.55 * v
        shape = np.exp(-(uu / wv) ** 2 * 2.2)
        top = sstep(hj + 0.02, hj - 0.3, v)
        I = shape * top * sstep(0.0, 0.04, v)
        I = np.clip(I * 1.5 - (1 - n1) * (0.25 + 0.9 * v), 0, 1)
        col = apply_lut(I, FIRE_LUT) * (I[..., None] ** 0.6) * 1.6
        return cv2.GaussianBlur(col.astype(np.float32), (0, 0), 1.0)

    def _sparks(self, frame, t, M, s):
        age = t - self.sp_t0
        alive = (age > 0) & (age < self.sp_life)
        if not alive.any():
            return
        a = age[alive]
        x = self.sp_x0[alive] + self.sp_vx[alive] * a + 25 * np.sin(a * 3 + self.sp_ph[alive])
        y = self.sp_y0[alive] + self.sp_vy[alive] * a + 40 * a * a
        b = self.sp_b[alive] * (1 - a / self.sp_life[alive]) * (0.6 + 0.4 * np.sin(a * 25 + self.sp_ph[alive]))
        P = apply_pts(M, np.stack([x, y], 1))
        ok = (P[:, 0] >= 1) & (P[:, 0] < OW - 1) & (P[:, 1] >= 1) & (P[:, 1] < OH - 1)
        if not ok.any():
            return
        P, b = P[ok], b[ok]
        buf = np.zeros((OH // 2, OW // 2), np.float32)
        np.add.at(buf, ((P[:, 1] / 2).astype(int), (P[:, 0] / 2).astype(int)), b * 3.0)
        buf = cv2.GaussianBlur(buf, (0, 0), max(0.7, s * 0.6)) * 2 + cv2.GaussianBlur(buf, (0, 0), 4) * 1.5
        buf = cv2.resize(buf, (OW, OH))
        frame += buf[..., None] * np.array([1.0, 0.55, 0.15])

    def _floor_fog(self, frame, t, M):
        # fog band on the stage floor (photo y > 1450) drifting sideways
        y = apply_pts(M, [[0, 1480], [0, 1832]])[:, 1]
        if y[0] > OH:
            return
        fogm = sstep(y[0], y[1], self.gy[:, :1])
        hz = cv2.resize(np.roll(self.n_haze, int(t * 45), axis=1)[:, :256], (OW, OH))
        frame += (fogm * hz)[..., None] * np.array([0.55, 0.4, 0.3]) * 0.3

    # ---------------- overlays
    def fg_crowd(self, frame, t, amount):
        if amount <= 0.01:
            return
        bp = 60.0 / max(1e-3, self.tl.get('bpm', 130))
        beat = pulse_train(self.beats, t, 0.15)
        for strip, base, i in self.fg_strips:
            sh, sw = strip.shape[:2]
            phase = (t / bp + i * 0.33) % 1.0
            jump = math.sin(math.pi * phase) ** 2 * (16 + 14 * i)
            head_y = (1725, 1850)[i]
            y0 = int(head_y - base - jump + (1 - amount) * 500)
            x0 = int(-100 + math.sin(t * 0.7 + i) * 30)
            ya, yb = max(0, y0), min(OH, y0 + sh)
            if ya >= yb:
                continue
            S = strip[ya - y0:yb - y0, -x0:-x0 + OW]
            a = S[..., 3:4] * amount
            rgb = S[..., :3] * (0.8 + 0.7 * self.hit(t) + 0.3 * beat)
            d = frame[ya:yb]
            d *= (1 - a)
            d += rgb * a

    def subtitles(self, frame, t):
        words = self.tl['words']
        chunks = self.tl['chunks']
        cur = None
        for c in chunks:
            if c['t0'] - 0.05 <= t < c['t1']:
                cur = c
        if cur is None:
            return
        ws = [words[i] for i in cur['words']]
        hi_col = (255, 176, 40) if cur['singer'] == 'dad' else ((130, 225, 255) if cur['singer'] == 'son' else (255, 230, 90))
        imgs = []
        for w in ws:
            active = w['t0'] - 0.03 <= t < w['t1'] + 0.08
            past = t >= w['t1'] + 0.08
            imgs.append((w, self.text.word(w['text'], hi_col if active else (255, 255, 255)), active, past))
        # layout into rows (max width)
        maxw = 900
        rows, row, rw = [], [], 0
        for it in imgs:
            wpx = it[1].shape[1] - 30
            if row and rw + wpx > maxw:
                rows.append((row, rw))
                row, rw = [], 0
            row.append(it)
            rw += wpx
        if row:
            rows.append((row, rw))
        app = t - cur['t0']
        pop = 1.0 + 0.22 * math.exp(-max(0, app) / 0.06) * math.cos(max(0, app) * 30) if app < 0.4 else 1.0
        fade = min(1.0, (cur['t1'] - t) / 0.06) if t > cur['t1'] - 0.06 else 1.0
        cy0 = self.tl.get('sub_y', 1210) - (len(rows) - 1) * 62
        for ri, (row, rw) in enumerate(rows):
            scs = [pop * (1.12 if active else 1.0) for (_, _, active, _) in row]
            wps = [(img.shape[1] - 30) * sc + 6 for (_, img, _, _), sc in zip(row, scs)]
            x = 520 - sum(wps) / 2
            y = cy0 + ri * 128
            for (w, img, active, past), sc, wpx in zip(row, scs, wps):
                jy = -10 if active else 0
                paste_rgba(frame, img, x + wpx / 2, y + jy, sc, fade)
                x += wpx

    def logo(self, frame, t):
        for L in self.tl.get('logos', []):
            if L['t0'] <= t < L['t1']:
                img = self.text.logo(L.get('text', 'THE BRAID BOYS'), L.get('size', 150))
                u = t - L['t0']
                sc = 1 + 1.2 * math.exp(-u / 0.07) if L.get('slam') else 1.0 + 0.04 * u
                al = min(1.0, u / 0.05) * min(1.0, (L['t1'] - t) / 0.15)
                paste_rgba(frame, img, OW / 2, L.get('y', 760), sc * L.get('scale', 1.0), al)
                if L.get('tag'):
                    tg = self.text.small(L['tag'], (255, 236, 200))
                    paste_rgba(frame, tg, OW / 2, L.get('y', 760) + 150, 1.0, al * min(1.0, max(0.0, (u - 0.25) / 0.2)))

    # ---------------- transitions + grade
    def transitions(self, frame, t):
        fl = 0.0
        for tr in self.tl.get('flashes', []):
            d = t - tr['t']
            if -0.04 <= d < 0.5:
                fl = max(fl, tr.get('a', 1.0) * (math.exp(-max(d, 0) / tr.get('decay', 0.08)) if d >= 0 else 1 + d / 0.04))
        for tr in self.tl.get('fireballs', []):
            d = t - tr['t']
            if -0.2 <= d < 0.45:
                self._fireball(frame, t, d, tr)
        if fl > 0.0:
            frame += fl * np.array([1.0, 0.92, 0.8], np.float32)
        st = pulse_train(self.strobes, t, 0.035) if len(self.strobes) else 0
        if st > 0.05:
            frame += st * 0.45

    def _fireball(self, frame, t, d, tr):
        # explosion that engulfs the frame by the cut (d=0) and burns away after it
        cx, cy = tr.get('x', OW / 2) / 4, tr.get('y', OH * 0.55) / 4
        r = np.hypot(self.qx - cx, self.qy - cy) / (QH * 0.5)
        ang = np.arctan2(self.qy - cy, self.qx - cx)
        if d < 0:
            g = (d + 0.2) / 0.2
            front = 0.05 + 1.5 * (1 - (1 - g) ** 2)
        else:
            front = 1.55 + d * 2.0
        N = self.n_fire[2]
        N2 = self.n_fire[0]
        ii = ((ang / (2 * math.pi) + 0.5) * 127).astype(int) % 128
        jj = ((r * 70 - t * 260) % 256).astype(int)
        n = 0.6 * N[jj, ii] + 0.4 * N2[(jj * 2) % 256, (ii * 3) % 128]
        body = sstep(front + 0.12, front - 0.3, r)
        burn = 0.0 if d < 0 else fsstep(0.0, 0.38, d)
        heat = 1.0 - 0.35 * np.clip(r / max(front, 1e-3), 0, 1)
        I = np.clip(body * heat * (0.55 + 0.75 * n) - burn * (1.35 - n) * 1.25, 0, 1)
        col = apply_lut(I, FIRE_LUT) * 1.35
        col = cv2.resize(cv2.GaussianBlur(col, (0, 0), 1.2), (OW, OH))
        occ = cv2.resize(np.clip(I * 1.4, 0, 1), (OW, OH))[..., None]
        frame *= (1 - 0.8 * occ)
        frame += col

    def grade(self, frame, t, fi, cam):
        hit = self.hit(t, 0.25)
        if cam.get('whip', 0) > 0.02:
            k = int(cam['whip'] * 140) | 1
            frame = cv2.blur(frame, (k, 1))
        # tone: filmic shoulder + contrast + warm highlights / teal shadows
        frame = np.clip(frame, 0, None)
        frame = frame / (1 + 0.28 * frame)
        frame *= 1.24
        lum = frame.mean(axis=2, keepdims=True)
        frame = lum + (frame - lum) * 1.12
        frame += (0.5 - np.clip(lum, 0, 1)) * np.array([-0.012, 0.004, 0.022], np.float32)
        frame = (frame - 0.5) * 1.07 + 0.5
        frame *= self.vignette
        # chromatic aberration on hits
        ca = int(round(7 * hit))
        if ca > 0:
            frame[..., 0] = np.roll(frame[..., 0], ca, axis=1)
            frame[..., 2] = np.roll(frame[..., 2], -ca, axis=1)
        g = cv2.resize(self.grain[fi % len(self.grain)], (OW, OH), interpolation=cv2.INTER_NEAREST)
        frame += g[..., None] * 0.022
        return frame

    # ---------------- main
    def render(self, fi):
        t = fi / self.fps
        s = self.shot_at(t)
        cam = self.camera(s, t)
        if s.get('kind') == 'arena':
            frame = self.arena(t, cam)
        elif s.get('kind') == 'image':
            frame = self.image_shot(t, s, cam)
        else:
            frame = self.stage(t, cam)
        self.fg_crowd(frame, t, s.get('fg_crowd', 0.0))
        self.transitions(frame, t)
        frame = self.grade(frame, t, fi, cam)
        self.logo(frame, t)
        self.subtitles(frame, t)
        return (np.clip(frame, 0, 1) * 255 + 0.5).astype(np.uint8)


def worker(args):
    D, tlp, font_dir, f0, f1, out = args
    tl = json.load(open(tlp))
    R = Renderer(D, tl, font_dir)
    cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{OW}x{OH}', '-r', str(tl['fps']),
           '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '15', '-pix_fmt', 'yuv420p', '-g', '30', out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        p.stdin.write(R.render(fi).tobytes())
        if (fi - f0) % 60 == 0:
            print(f'  seg {f0}: frame {fi}', flush=True)
    p.stdin.close()
    p.wait()
    return out


def main():
    D, tlp, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    font_dir = os.path.join(os.path.dirname(D.rstrip('/')), 'fonts')
    tl = json.load(open(tlp))
    if mode == 'frames':
        out = sys.argv[4]
        os.makedirs(out, exist_ok=True)
        R = Renderer(D, tl, font_dir)
        import time
        for ts in sys.argv[5].split(','):
            t0 = time.time()
            fi = int(round(float(ts) * tl['fps']))
            img = R.render(fi)
            Image.fromarray(img).save(f'{out}/f_{float(ts):05.2f}.jpg', quality=90)
            print(ts, round(time.time() - t0, 2), 's', flush=True)
    else:
        outp, audio = sys.argv[4], sys.argv[5]
        nw = int(sys.argv[6]) if len(sys.argv) > 6 else 4
        nf = int(round(tl['duration'] * tl['fps']))
        bounds = np.linspace(0, nf, nw + 1).astype(int)
        tmpd = outp + '_segs'
        os.makedirs(tmpd, exist_ok=True)
        jobs = [(D, tlp, font_dir, int(bounds[i]), int(bounds[i + 1]), f'{tmpd}/seg{i:02d}.mp4') for i in range(nw)]
        from multiprocessing import Pool
        with Pool(nw) as pool:
            segs = pool.map(worker, jobs)
        with open(f'{tmpd}/list.txt', 'w') as f:
            for sg in segs:
                f.write(f"file '{os.path.abspath(sg)}'\n")
        subprocess.check_call(['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', f'{tmpd}/list.txt', '-i', audio,
                               '-map', '0:v', '-map', '1:a', '-c:v', 'copy', '-c:a', 'aac', '-b:a', '320k', '-ar', '48000',
                               '-shortest', '-movflags', '+faststart', outp])
        print('wrote', outp)


if __name__ == '__main__':
    main()
