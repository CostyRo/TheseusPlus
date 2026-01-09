# TheseusPlus

TheseusPlus is a modernized version of the original Theseus web app. This project was created to update the software, keep it working with modern Python/Dash, and support newer CPD (change point detection) algorithms.

## Setup

Requirements: Python 3.9+ and `uv`.

```bash
python -m pip install uv
uv sync
uv run python app.py
```

Open http://127.0.0.1:8050/ in your browser.

## Data

This repo includes a small subset of the benchmark under `data/` so the app can run out of the box.
For the full datasets, download and unzip:

- https://www.thedatum.org/datasets/TSB-UAD-Public.zip

## Credits

Original Theseus authors:

- Paul Boniol
- John Paparrizos
