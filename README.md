# Real-Time Dynamic Direction-of-Arrival Tracking via SDR Using Multiple Bayesian Filters

This repository contains the EE4084 project implementation for real-time dynamic Direction-of-Arrival (DoA) tracking with ADALM-Pluto SDR devices. The system transmits a BPSK frame, receives it with a two-channel SDR receiver, estimates the incoming angle from the inter-channel phase difference, and smooths the raw DoA measurements using multiple Bayesian tracking filters.

## Team Members

- Kerem Ergünay - 150721039
- Mercan Tuana Polat - 150721072

## Project Presentation

- YouTube Presentation Link: [https://youtu.be/W27EqYjI83k](https://youtu.be/W27EqYjI83k)

## Project Overview

The project focuses on practical real-time DoA tracking under hardware noise, carrier frequency offset, antenna phase mismatch, and multipath effects. A PyQt5 dashboard displays raw and filtered angle estimates, link quality metrics, decoded BPSK payload data, calibration state, and confidence information during live experiments.

Implemented tracking methods:

- g-h filter
- 1D Kalman filter
- Particle filter
- Interacting Multiple Model (IMM) Kalman filter

## Repository Structure

| Path | Description |
|---|---|
| `Beamforming/Codes/Kalman_Proje.py` | Main PyQt5 GUI, SDR control, synchronization, DoA estimation, and filter implementation |
| `Beamforming/Codes/run_kalman_usb.sh` | Launcher script that resolves PlutoSDR USB libiio URIs by device serial number |
| `Beamforming/Codes/kalman_proje_state.json` | Default calibration and antenna-spacing state |
| `report_latex/` | Complete LaTeX report source, bibliography, and figures |
| `requirements.txt` | Python package dependencies |

## Hardware Configuration

- SDR hardware: two ADALM-Pluto SDR devices
- Transmitter: one PlutoSDR transmitting the BPSK frame
- Receiver: one PlutoSDR using two receive channels for phase-difference DoA estimation
- Carrier frequency: `1.3 GHz`
- Sampling rate: `2 MS/s`
- Antenna spacing: `lambda/2`, approximately `11.54 cm`
- Displayed DoA range: `-45 deg` to `+45 deg`
- Default transmitted message: `KALMAN_PROJE`

## Software Requirements

Install the Python dependencies:

```bash
python -m venv venv
venv/bin/python -m pip install -r requirements.txt
```

System dependency:

- `libiio` must be installed.
- `iio_info -s` should list both connected PlutoSDR devices.

## Running the System

Start the GUI and resolve the PlutoSDR USB contexts automatically:

```bash
Beamforming/Codes/run_kalman_usb.sh --swap
```

If the displayed angle sign is reversed, run:

```bash
Beamforming/Codes/run_kalman_usb.sh --swap --invert
```

Useful diagnostic option:

```bash
Beamforming/Codes/run_kalman_usb.sh --swap --print-uris
```

## Calibration Procedure

1. Place the transmitter at boresight (`0 deg`).
2. Start the GUI and wait for Barker synchronization lock.
3. Press `Calibrate (Zero)` to store the boresight phase offset.
4. Move the transmitter and compare the raw and filtered DoA traces.

## LaTeX Report

The complete report source is available in `report_latex/`.

Build command sequence:

```bash
cd report_latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Notes

- Runtime calibration files generated for specific SDR serial-number pairs are ignored by Git.
- Large IQ recordings and generated build artifacts are intentionally excluded from this repository.
