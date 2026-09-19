# Traffic Signal System Using Python

A desktop traffic-analysis application that detects vehicles in images, videos, and live camera feeds using YOLO. It calculates vehicle density, classifies traffic level, estimates signal green time, stores traffic history, and exports CSV reports.

## Features

- Vehicle detection for cars, motorcycles, buses, and trucks
- YOLO-based tracking for video and webcam input
- Optional road or lane region of interest (ROI)
- Traffic density and traffic-level calculation
- Signal green-time recommendation
- Traffic history dashboard
- CSV report export
- Tkinter desktop interface

## Requirements

- Python 3.9 or newer
- Webcam or video file for video analysis
- `yolov8n.pt` model file in the `models` directory

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download or copy the YOLO model file to:

```text
models/yolov8n.pt
```

## Run the Application

From the project root:

```powershell
python app.py
```

Use the interface to upload an image, open a video, start a webcam stream, select an optional road region, detect vehicles, view the dashboard, or generate a CSV report.

## Project Structure

```text
TRAFFIC_CS/
├── app.py                  # Application entry point
├── config.py               # Paths, model, vehicle classes, and signal rules
├── requirements.txt        # Python dependencies
├── models/                 # YOLO model files
├── src/
│   ├── detector.py         # Vehicle detection and tracking
│   ├── history.py          # History storage and report export
│   ├── roi.py              # Region-of-interest selection and cropping
│   ├── traffic.py          # Density and traffic calculations
│   └── video_srvice.py     # Video and webcam processing loop
├── ui/
│   └── main_window.py      # Tkinter user interface
├── data/                   # Local input and history data
├── outputs/                # Generated detection images and reports
└── tests/                  # Automated tests
```

## Data and Outputs

The application creates required directories automatically. Local images, videos, history files, generated reports, and Python cache files are excluded from Git by `.gitignore`.

## Testing

Run the test suite with:

```powershell
python -m pytest -q
```

Run a syntax check for all Python files with:

```powershell
python -m compileall -q .
```

## Notes

- The first YOLO model load may take a moment.
- Webcam access depends on operating-system permissions and camera availability.
- Detection accuracy depends on the selected model and input quality.
