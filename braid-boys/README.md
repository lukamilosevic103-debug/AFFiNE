# The Braid Boys: "Carry My Name" (30 s TikTok)

**Finished video:** `the_braid_boys_tiktok.mp4` (1080×1920, 9:16, 30 fps, 30.0 s, H.264 + AAC 320 kbps, −10 LUFS)

## Structure

| Time | Section | What happens |
|---|---|---|
| 0.0–1.8 s | Hook | Logo slam "THE BRAID BOYS / LIVE IN CONCERT", pyro, crowd roar |
| 1.8–7.4 s | Verse (dad) | "I FOUGHT THROUGH HELL, I CONQUERED THE FLAME! / NOW RISE, MY SON, AND CARRY MY NAME!" |
| 7.4 s | Drop | Full-screen fireball, flash and sub boom as the band kicks in |
| 7.4–14.4 s | Chorus (son) | "I FEEL YOUR FIRE RUNNING THROUGH ME! / THE BEAST YOU ARE IS THE BEAST I'LL BE!" with an eye glint on "BEAST" |
| 14.4–18.6 s | Break | Packed-arena cutaways, crowd "HEY! HEY!" chant, strobes |
| 18.6–25.5 s | Chorus (both) | Son lead with dad an octave below, strobes on every beat |
| 25.5–30 s | Outro | "THE BEAST I'LL BE!", final pyro, logo and "FOLLOW FOR PART 2" |

The subtitles are word-by-word karaoke: orange for the dad, ice-blue for the son and gold for both together.

## How this version was made (all local, no paid services)

- **Band:** MusicGen stereo-medium, from the prompt "epic stadium hard rock anthem…" (139 BPM, E). It builds for 7 s, then drops.
- **Vocals:** Bark (Suno's open-source voice model) generated the takes, and Whisper aligned each word. Each word was then
  re-timed onto the beat grid and tuned to a melody with the WORLD vocoder. The dad's voice is deepened, and the son's has
  child-like formants. Both are double-tracked and run through a rock vocal chain with arena reverb.
- **Crowd:** synthesized roar, cheers, applause and a "HEY!" chant, plus pyro whooshes and booms.
- **Video:** a 2.5D composite of the band photo with people masks (Mask2Former), virtual camera moves, living
  flames, fire jets, light beams, sparks, floor fog, beat-synced bounce, vocal-driven jaw motion, and eye glints.
  Crowd b-roll comes from SDXL-Lightning (`broll/`).

`audio/` holds the full mix plus the band, dad and son stems. `pipeline/` holds the scripts and configs used to render.

## Upgrading to Suno + Kling + Hedra

See `PRODUCTION_KIT.md` for copy-paste prompts, a shot list and an edit timeline. Swapping in a Suno vocal and
Hedra or Kling clips will mainly improve the singing realism (real lip-sync and a real singer) and the body movement.
