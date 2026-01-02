import essentia.standard as es
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import butter, filtfilt

# 1. LOAD AUDIO


audio_path = "Nachna.wav"
sr = 44100
hop = 256

audio = es.MonoLoader(filename=audio_path, sampleRate=sr)()


# 2. MELODY EXTRACTION (MELODIA)


melody_extractor = es.PredominantPitchMelodia(
    frameSize=2048,
    hopSize=hop,
    sampleRate=sr,
    minFrequency=55,
    maxFrequency=900,
    guessUnvoiced=True
)

pitch, confidence = melody_extractor(audio)
pitch[pitch == 0] = np.nan


# 3. MAKE MELODY WHISTLE-LIKE (CONTROL-RATE)


def limit_pitch_speed(pitch, sr, hop, max_hz_per_sec=600):
    out = pitch.copy()
    out[np.isnan(out)] = 0.0
    max_step = max_hz_per_sec * (hop / sr)

    for i in range(1, len(out)):
        if out[i] == 0:
            out[i] = out[i - 1]
            continue
        delta = out[i] - out[i - 1]
        if abs(delta) > max_step:
            out[i] = out[i - 1] + np.sign(delta) * max_step
    return out

def slow_control(pitch, factor=6):
    reduced = pitch[::factor]
    x_old = np.linspace(0, 1, len(reduced))
    x_new = np.linspace(0, 1, len(pitch))
    return np.interp(x_new, x_old, reduced)

pitch = limit_pitch_speed(pitch, sr, hop)
pitch = slow_control(pitch, factor=6)

# Vibrato (very subtle)
t_frames = np.arange(len(pitch)) * hop / sr
pitch += 5 * np.sin(2 * np.pi * 5 * t_frames)


# 4. CONTROL → AUDIO RATE


duration = len(pitch) * hop / sr
t_audio = np.arange(int(duration * sr)) / sr

interp_pitch = np.interp(
    np.linspace(0, len(pitch), len(t_audio)),
    np.arange(len(pitch)),
    np.nan_to_num(pitch, nan=np.nanmean(pitch))
)

# 5. WHISTLE CORE (PSYCHOACOUSTIC)


phase1 = np.cumsum(interp_pitch) / sr
phase2 = np.cumsum(2 * interp_pitch) / sr
phase3 = np.cumsum(3 * interp_pitch) / sr

core = np.zeros_like(interp_pitch)

for i, f in enumerate(interp_pitch):
    if f < 300:
        core[i] = (
            0.15 * np.sin(2 * np.pi * phase1[i]) +
            0.55 * np.sin(2 * np.pi * phase2[i]) +
            0.30 * np.sin(2 * np.pi * phase3[i])
        )
    elif f < 600:
        core[i] = (
            0.30 * np.sin(2 * np.pi * phase1[i]) +
            0.50 * np.sin(2 * np.pi * phase2[i]) +
            0.20 * np.sin(2 * np.pi * phase3[i])
        )
    else:
        core[i] = np.sin(2 * np.pi * phase1[i])


# 6. BREATH NOISE (SHAPED, NOT RAW)


noise = np.random.randn(len(core))
noise = np.convolve(noise, np.ones(300) / 300, mode="same")

whistle_noise = np.zeros_like(noise)
nyq = sr / 2
bandwidth = 400  # narrow jet

for i in range(0, len(noise), 1024):
    c = interp_pitch[min(i, len(interp_pitch) - 1)]
    low = max(300, c - bandwidth / 2)
    high = min(nyq - 300, c + bandwidth / 2)

    if low >= high:
        whistle_noise[i:i+1024] = noise[i:i+1024]
        continue

    b, a = butter(2, [low / nyq, high / nyq], btype="band")
    whistle_noise[i:i+1024] = filtfilt(b, a, noise[i:i+1024])


# 7. MIX CORE + AIR


whistle = 0.6 * core + 0.4 * whistle_noise

# Remove muddy low frequencies
b_hp, a_hp = butter(1, 300 / nyq, btype="high")
whistle = filtfilt(b_hp, a_hp, whistle)


# 8. AMPLITUDE ENVELOPE (BREATH)


amp = np.abs(np.gradient(interp_pitch))
amp = 1 - amp / (np.max(amp) + 1e-6)
amp = np.convolve(amp, np.ones(1000) / 1000, mode="same")

whistle *= (0.6 + 0.4 * amp)

# Subtle airflow instability
air = np.random.randn(len(whistle))
air = np.convolve(air, np.ones(2000) / 2000, mode="same")
whistle *= (1 + 0.02 * air)


# 9. FINAL TONE POLISH


b_lp, a_lp = butter(1, 3500 / nyq)
whistle = filtfilt(b_lp, a_lp, whistle)

# Normalize
whistle /= np.max(np.abs(whistle) + 1e-6)


# 10. SAVE + OPTIONAL PLOT


sf.write("whistle_output.wav", whistle, sr)
print("Saved whistle_output.wav")

plt.figure(figsize=(12, 4))
plt.plot(t_frames, pitch)
plt.title("Final Whistle Pitch Control")
plt.xlabel("Time (s)")
plt.ylabel("Hz")
plt.grid(True)
plt.tight_layout()
plt.show()
