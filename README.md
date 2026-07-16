# AI-Powered Audio Deepfake and Voice Clone Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An enterprise-grade, decoupled AI security solution designed to authenticate human speech and detect highly sophisticated generative AI voice clones and audio deepfakes. Developed as a Mini Project (**BCS-554**) at the Department of Computer Science & Engineering, Pranveer Singh Institute of Technology (PSIT), Kanpur.

---

## 📌 Executive Summary

The rapid rise of text-to-speech (TTS) and voice conversion models has compromised traditional audio biometrics, giving rise to highly convincing social engineering and financial fraud.

To address this threat reliably without application freezing or Out-of-Memory (OOM) crashes, this project adopts a **decoupled, multi-threaded client-server architecture**:

1. **The Backend (FastAPI):** An asynchronous, high-performance API server. It loads our deep learning models into memory *once* upon startup and exposes lightning-fast inference endpoints.
2. **The Frontend (HTML5/CSS3/JavaScript):** A lightweight, responsive web dashboard that offloads all heavy processing to the backend, preventing UI freezing and ensuring high-concurrency stability.

By transforming raw audio signals into 2D **Mel-Spectrograms** using Digital Signal Processing (DSP), we treat deepfake voice detection as a **Computer Vision task**. A custom **Convolutional Neural Network (CNN)** parses these spectrograms to detect sub-perceptual synthetic vocoder signatures, unnatural silences, and phase anomalies invisible to the human ear.

---

## ⚙️ Decoupled System Architecture

Unlike monolithic web apps where the UI and machine learning execution compete for Python's single thread, our architecture keeps them completely independent.

```text
  ┌──────────────────────────────┐
  │     Lightweight UI Client    │
  │   (HTML5 / CSS3 / Vanilla JS)│
  └──────────────┬───────────────┘
                 │
                 │ 1. HTTP POST /predict (Audio File)
                 ▼
  ┌──────────────────────────────┐
  │    Asynchronous FastAPI      │ ◄── [ Loads PyTorch CNN Once ]
  └──────────────┬───────────────┘
                 │
                 │ 2. Runs Librosa DSP (Mel-Spectrogram)
                 │ 3. Forward pass through 2D CNN
                 ▼
  ┌──────────────────────────────┐
  │     JSON Response Object     │ ──► {"verdict": "FAKE", "confidence": 0.982}
  └──────────────────────────────┘
