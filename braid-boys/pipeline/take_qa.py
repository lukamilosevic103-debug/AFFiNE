import sys, numpy as np, librosa, warnings
warnings.filterwarnings('ignore')
from faster_whisper import WhisperModel
m = WhisperModel('small.en', device='cpu', compute_type='int8', cpu_threads=2)
for f in sys.argv[1:]:
    y, sr = librosa.load(f, sr=16000)
    segs, _ = m.transcribe(y.astype(np.float32), word_timestamps=True, beam_size=5, language='en')
    words = [(w.word.strip(), round(w.start, 2), round(w.end, 2)) for s in segs for w in s.words]
    f0, vf, vp = librosa.pyin(y, fmin=60, fmax=900, sr=16000, frame_length=1024)
    v = f0[~np.isnan(f0)]
    rms = librosa.feature.rms(y=y)[0]
    print(f.split('/')[-1], f'dur {len(y)/16000:.2f}s voiced {len(v)/len(f0):.2f} f0 med {np.median(v) if len(v) else 0:.0f}Hz p10-90 {np.percentile(v,10) if len(v) else 0:.0f}-{np.percentile(v,90) if len(v) else 0:.0f} peak {np.abs(y).max():.2f}')
    print('   ', ' '.join(f'{w}[{a}-{b}]' for w, a, b in words))
