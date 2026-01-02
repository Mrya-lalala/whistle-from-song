# Whistle From Song  
(DSP Prototype / Exploratory Project)

This repository contains an experimental attempt to convert a song’s melody into a whistle-like sound using signal processing techniques.

This project documents exploration, limitations, and conclusions reached using a pure DSP approach. A data-driven (machine learning) approach is planned as a follow-up.

---

## Motivation

The goal of this project was to explore whether it is possible to take the melody of any song and render it as a smooth, human-like whistle using classical signal processing methods alone.

This repository represents the first phase of that exploration.

---

## What This Project Does

- Loads a monophonic audio signal
- Extracts the predominant melody using Essentia’s Melodia algorithm
- Post-processes the pitch contour to make it smoother and more human-like
- Attempts to synthesize a whistle-like sound using oscillators, noise, envelopes, and filtering
- Evaluates how far DSP techniques alone can approximate human whistling

---

## Repository Structure

whistle-from-song/
├── .gitignore
├── README.md
├── whistle_from_song.py
└── requirements.txt

### Key File

- `whistle_from_song.py`  
  Complete DSP prototype that:
  - loads audio
  - extracts melody using Melodia
  - smooths the pitch contour
  - synthesizes a whistle-like sound

---

## Dependencies

Python packages:
- numpy
- scipy
- librosa
- soundfile
- matplotlib
- essentia

Essentia installation on macOS:
bash
brew install essentia

How to run:
pip install -r requirements.txt
ffmpeg -i song.mp3 -ac 1 -ar 44100 song.wav
audio_path = "song.wav"
python whistle_from_song.py

