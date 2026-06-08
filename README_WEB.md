# Real-time Violence Detection - Web App (Flask)

## Quickstart

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Ensure model weights exist

- File required: `mobilenetv2_violence.weights.h5` in the project root.

3) Run the web app

```bash
python app.py
```

4) Open the UI

- Visit http://localhost:5000
- Upload an MP4/AVI/MOV/MKV video
- Wait for processing to complete
- Download the processed video and CSV report

## Features
- Upload preview before processing
- Real-time status with progress bar
- Frame-by-frame inference with labels
- Timeline of violent segments
- Download processed video and CSV summary

## Notes
- Outputs are stored in `web_outputs/` and `reports/`.
- Uploads are stored in `uploads/`.
- The detection threshold is 0.5; adjust in `app.py` if needed.


