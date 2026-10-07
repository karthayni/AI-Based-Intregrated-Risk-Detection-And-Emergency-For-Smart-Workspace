# Module 1: PPE Detection and High-Risk Alert

Part of the project **AI-Based Integrated Safety And Risk System For Smart Workspace**.

The module detects workers and their PPE (helmet, vest, boots) from a webcam or video and classifies each worker as
**Safe**, **Violation** or **High-Risk** (PPE missing + inside a danger zone). Problems are saved to `logs/events.csv` with a snapshot.

## How it works
1. `yolov8n.pt` (standard model) finds people.
2. `best.pt` (our YOLOv8s model trained on a Roboflow PPE dataset) finds helmet, vest, boots and gloves.
3. PPE is matched to each person, the danger zone is checked, and the status is shown and logged.

## Results (validation, mAP50)
Vest 96.6%, Helmet 96.4%, Boots 94.1%, Gloves 28.4% (needs more data).

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```
python ppe_detect.py --source 0 --draw-zone                  # mark the danger zone once
python ppe_detect.py --weights best.pt --source 0            # run detection (q = quit, f = full screen)
python ppe_detect.py --weights best.pt --source 0 --fullscreen
python ppe_detect.py --weights best.pt --source 0 --require helmet vest shoes
```

## Training
`train.py` shows the training command. We trained on Google Colab (Tesla T4, about 0.89 hours) with the Roboflow dataset
`ppe-detection-qlq3d` (CC BY 4.0, 5,140 images).

## Files
- `ppe_detect.py`: real-time detection, danger zone, status and alert logging
- `train.py`, `data.yaml`: training script and class list
- `best.pt`: trained PPE model (not included if you download it from Releases)
