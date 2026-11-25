# The Braid Boys: "Carry My Name" (30 s TikTok production kit)

Reference format: dad and son on stage, the son takes the chorus, the crowd goes wild, and the dad
looks on proudly (like the Iron Peak "One Day, Little Viking" clip). Here it's a heavier rock version
with fire, an arena crowd and new lyrics.

---

## 1. Suno: the song

**Song title:** Carry My Name

**Style of music** (paste into "Styles"):

```
arena hard rock anthem, 140 bpm, E major, stomping drums, huge distorted power chords, gang vocals,
duet: deep gritty baritone male vocal (father) and an energetic young boy vocal (child, son),
live stadium energy, crowd chants, pyro-fueled, catchy shout-along chorus, polished modern mix
```

**Exclude styles:** `female vocals, autotune, EDM, trap, lo-fi`

**Lyrics** (paste into "Lyrics"):

```
[Intro: crowd roar, drum build]

[Verse 1: Father, deep powerful baritone]
I FOUGHT THROUGH HELL, I CONQUERED THE FLAME!
NOW RISE, MY SON, AND CARRY MY NAME!

[Drop]
[Chorus: Son, young boy voice, explosive energy]
I FEEL YOUR FIRE RUNNING THROUGH ME!
THE BEAST YOU ARE IS THE BEAST I'LL BE!

[Break: crowd chanting "HEY! HEY!"]

[Chorus: Father and Son together, gang vocals]
I FEEL YOUR FIRE RUNNING THROUGH ME!
THE BEAST YOU ARE IS THE BEAST I'LL BE!

[Outro: Son and Father]
THE BEAST I'LL BE!
[Big ending, crowd roar]
```

**Tips**

- Generate 4–6 versions. Pick the one where the chorus kid voice really sounds like a child.
- If the son sounds too adult, open the song → **Edit → Replace Section** on the chorus. Re-describe it as
  `young boy vocal, child singer, high bright voice, shouting with joy`.
- Download **stems** (vocals and instrumental). Hedra lip-sync works best with the isolated vocal of each singer.
- Cut the final edit to the first ~30 s: verse → drop → chorus.

---

## 2. Hedra (Character-3): singing faces with real lip-sync

Make **two crops** from the band photo:

- **Dad:** head and chest, mic visible, 9:16
- **Son:** head to waist, fists visible, 9:16

| Clip | Image | Audio | Prompt |
|---|---|---|---|
| Dad verse | dad crop | dad vocal 0:02–0:08 | `intense rock singer belting into the microphone, veins in neck, braids swinging, stage fire flickering behind, sweat, powerful` |
| Son chorus | son crop | son vocal 0:07–0:14 | `young boy rock singer shouting the chorus with fierce joy, fists clenched, head banging, bright blue eyes, stage lights strobing` |
| Duet | full photo | mixed vocals 0:18–0:25 | `father and son singing together on a fire-lit arena stage, both energetic, crowd lights` |

Export at 9:16, 30 fps, highest quality.

---

## 3. Kling AI: concert motion, crowd and b-roll (image-to-video)

Use the band photo as the **start frame**, with **Professional** mode, 9:16, and 5 s or 10 s clips. Keep
`Creativity` low (≈0.3) so the faces and outfits stay consistent. Use the same photo as the **element/reference** in every clip.

| # | Clip | Kling prompt |
|---|---|---|
| K1 | Opening wide | `Slow dolly push-in toward the stage, four pyrotechnic flame columns erupt upward, sparks rain down, stage lights strobe, father and son stand ready, massive arena crowd cheering with raised hands in the foreground, haze, 4k concert footage` |
| K2 | Dad hype | `The father sings powerfully into the microphone and stomps forward, braids swinging, flames burst behind him, handheld camera shake, warm orange light` |
| K3 | Son rises | `The young boy steps forward, pumps his fists, then jumps on the beat; the camera whips from the father to the son; sparks fly` |
| K4 | Drop | `Huge fireball explosion fills the stage, camera shake, the boy screams into the crowd with arms wide, confetti and sparks, strobe lights` |
| K5 | Crowd reaction | `Reverse angle from the stage over a sold-out arena of 20,000 fans jumping and screaming, phone lights, hands in the air, light beams sweeping through smoke` |
| K6 | Proud dad | `The father stops singing and puts his hand on his son's shoulder, smiling proudly while the boy sings, flames behind them, slow motion` |
| K7 | Finale | `Both raise their fists as every flame cannon fires at once, crowd explodes, camera pulls back to reveal the whole arena, epic` |

Kling also has **Lip Sync**: drop the matching vocal onto K2/K3 if you skip Hedra.

---

## 4. Edit (CapCut / Premiere): 1080×1920, 30 fps

| Time | Section | Shots | FX |
|---|---|---|---|
| 0.0–1.8 s | Hook | K1 wide | logo slam "THE BRAID BOYS", flash, crowd roar |
| 1.8–7.4 s | Verse (dad) | Hedra dad → K3 whip to son → two-shot | pyro on "HELL" and "FLAME", whip pans |
| 7.4 s | **Drop** | K4 | fireball transition, white flash, camera shake, bass boom |
| 7.4–14.4 s | Chorus (son) | Hedra son, K6 proud dad | beat-synced zoom punches, eye glint on "BEAST" |
| 14.4–18.6 s | Break | K5 crowd × 2 | "HEY! HEY!" chant, strobes |
| 18.6–25.5 s | Chorus (both) | Hedra duet, quick cuts every 2 beats | strobes on every beat |
| 25.5–30.0 s | Outro | K7 | "THE BEAST I'LL BE!", final pyro, logo + "FOLLOW FOR PART 2" |

**Subtitles:** word-by-word karaoke in a heavy condensed font (Anton). Use a white fill with a thick black stroke, orange
for the dad's words, ice-blue for the son's and gold when both sing. Place them in the lower-middle third (y ≈ 1200 px) to stay clear of the
TikTok UI.

**Audio:** Suno master, plus a crowd-cheer layer (−14 dB under the vocals, up at the hook, break and ending) and a
fire "whoosh" on each pyro hit. Keep the master near −10 LUFS, peaking at −1 dB.

---

## 5. TikTok post

**Caption:** `He told his son to carry his name... the kid took the whole arena 🔥🐺 #TheBraidBoys`

**Hashtags:** `#TheBraidBoys #fatherandson #rock #metal #concert #fyp #viral #rockmusic #livemusic #aimusic`

Post with the in-app sound credited to **The Braid Boys: Carry My Name**, then pin a comment: "Part 2?".
