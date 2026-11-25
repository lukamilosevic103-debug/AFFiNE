import sys, numpy as np, librosa, warnings
warnings.filterwarnings('ignore')
for f in sys.argv[1:]:
    y, sr = librosa.load(f, sr=None, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, units='time')
    tempo = float(np.atleast_1d(tempo)[0])
    ibi = np.diff(beats)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1)
    # Krumhansl key estimate
    maj = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]); mnr = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    names = 'C C# D D# E F F# G G# A A# B'.split()
    sc = [(np.corrcoef(np.roll(maj,k),chroma)[0,1], names[k]+' major') for k in range(12)] + [(np.corrcoef(np.roll(mnr,k),chroma)[0,1], names[k]+' minor') for k in range(12)]
    key = max(sc)
    rms = librosa.feature.rms(y=y, frame_length=sr, hop_length=sr)[0]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    flat = librosa.feature.spectral_flatness(y=y).mean()
    print(f'{f.split("/")[-1]}: dur {len(y)/sr:.1f}s tempo {tempo:.1f} beats {len(beats)} ibi_std {ibi.std():.3f} first {beats[:3].round(2)} key {key[1]} ({key[0]:.2f}) centroid {cent:.0f} flat {flat:.3f}')
    print('   rms/sec:', ' '.join(f'{v:.2f}' for v in rms))
