"""real-time PPE detection + compliance check + danger zone + alerts.

Draw the danger zone once:   python ppe_detect.py --source 0 --draw-zone
Run detection:               python ppe_detect.py --weights runs/ppe/weights/best.pt --source 0
Run on a video file:         python ppe_detect.py --weights best.pt --source site.mp4
Press 'f' for full screen, 'q' to quit.
Bigger window:                python ppe_detect.py --weights best.pt --source 0 --fullscreen
"""
import argparse, csv, json, time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---- CONFIG -----------------------------------------------------------------
ALIASES = {                            # model class name (lower-case) -> PPE category
    "helmet": ["hardhat", "hard hat", "helmet"],
    "vest":   ["safety vest", "vest"],
    "gloves": ["gloves", "glove"],
    "shoes":  ["safety shoes", "boots", "boot", "shoes"],
}
ZONE_FILE = "zone.json"
LOG_DIR = Path("logs")
COLORS = {"SAFE": (60, 170, 60), "VIOLATION": (0, 190, 255), "HIGH_RISK": (50, 50, 230)}
RANK = {"SAFE": 0, "VIOLATION": 1, "HIGH_RISK": 2}
# -----------------------------------------------------------------------------


def category(name):
    """Map a model class name to a PPE category, or None (ignores 'NO-Hardhat' style classes)."""
    n = name.lower().strip()
    if n.startswith("no-") or n.startswith("no "):
        return None
    for cat, words in ALIASES.items():
        if n in words:
            return cat
    return None


def in_zone(point, zone):
    if zone is None or len(zone) < 3:
        return False
    return cv2.pointPolygonTest(np.array(zone, np.int32), (float(point[0]), float(point[1])), False) >= 0


def evaluate(persons, ppe, zone, required):
    """persons: [(x1,y1,x2,y2)], ppe: [(category,(x1,y1,x2,y2))] -> list of result dicts."""
    results = []
    for (x1, y1, x2, y2) in persons:
        h = max(y2 - y1, 1)
        found = set()
        for cat, (a1, b1, a2, b2) in ppe:
            cx, cy = (a1 + a2) / 2, (b1 + b2) / 2
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue                                   # PPE is not on this person
            if cat == "helmet" and cy > y1 + 0.45 * h:     # helmet must be near the head
                continue
            if cat == "shoes" and cy < y1 + 0.70 * h:      # shoes must be near the feet
                continue
            found.add(cat)
        missing = [r for r in required if r not in found]
        foot = ((x1 + x2) / 2, y2)                         # bottom-centre = where the person stands
        inside = in_zone(foot, zone)
        if not missing:
            status = "SAFE"
        elif inside:
            status = "HIGH_RISK"
        else:
            status = "VIOLATION"
        results.append(dict(box=(x1, y1, x2, y2), missing=missing, in_zone=inside, status=status))
    return results


def send_alert(status, details, snapshot):
    """Hook: replace the print with SMS / email / Telegram / dashboard API call."""
    print(f"[ALERT] {status}: {details}  -> {snapshot}")


def log_event(frame, status, details):
    (LOG_DIR / "snapshots").mkdir(parents=True, exist_ok=True)
    ts = datetime.now()
    snap = LOG_DIR / "snapshots" / f"{ts:%Y%m%d_%H%M%S}_{status}.jpg"
    cv2.imwrite(str(snap), frame)
    new = not (LOG_DIR / "events.csv").exists()
    with open(LOG_DIR / "events.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "status", "details", "snapshot"])
        w.writerow([ts.isoformat(timespec="seconds"), status, details, snap])
    send_alert(status, details, snap)


def draw_zone_tool(source, width=1280, height=720):
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):                       # same size as live detection, so the zone lines up
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(5):                                # let the camera settle
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Cannot read from source")
    pts = []

    def click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y])

    cv2.namedWindow("Draw danger zone")
    cv2.setMouseCallback("Draw danger zone", click)
    while True:
        img = frame.copy()
        for p in pts:
            cv2.circle(img, tuple(p), 5, (0, 0, 255), -1)
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], len(pts) > 2, (0, 0, 255), 2)
        cv2.putText(img, "Click corners | s = save | r = reset | q = cancel", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Draw danger zone", img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("r"):
            pts.clear()
        elif k == ord("s") and len(pts) >= 3:
            json.dump(pts, open(ZONE_FILE, "w"))
            print(f"Saved {len(pts)} points to {ZONE_FILE}")
            break
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/ppe/weights/best.pt")
    ap.add_argument("--source", default="0", help="0 = webcam, or path to a video file")
    ap.add_argument("--person-weights", default="yolov8n.pt",
                    help="model that finds people (your PPE dataset has no 'person' class, so we use the standard COCO model)")
    ap.add_argument("--require", nargs="+", default=["helmet", "vest"],
                    help="PPE that every person must wear, e.g. --require helmet vest gloves shoes")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--persist", type=int, default=5, help="frames a status must persist before it is logged")
    ap.add_argument("--cooldown", type=float, default=10, help="seconds between logged alerts")
    ap.add_argument("--draw-zone", action="store_true")
    ap.add_argument("--fullscreen", action="store_true", help="start in full screen (press f to toggle, q to quit)")
    ap.add_argument("--width", type=int, default=1280, help="camera width to request (bigger = sharper picture)")
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    source = int(a.source) if a.source.isdigit() else a.source

    if a.draw_zone:
        return draw_zone_tool(source, a.width, a.height)

    from ultralytics import YOLO
    model = YOLO(a.weights)                 # your trained PPE model (helmet, vest, gloves, boots...)
    person_model = YOLO(a.person_weights)   # standard model, class 0 = person
    zone = json.load(open(ZONE_FILE)) if Path(ZONE_FILE).exists() else None
    if zone is None:
        print("No zone.json found: danger-zone check disabled (run with --draw-zone first).")

    cap = cv2.VideoCapture(source)
    if isinstance(source, int):                       # ask the webcam for a larger picture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    WIN = "PPE Monitoring"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)           # resizable window (drag corners or maximize button)
    cv2.resizeWindow(WIN, 1280, 720)
    full = a.fullscreen
    if full:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    bad_frames, last_log = 0, 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        persons, ppe = [], []
        pres = person_model(frame, conf=a.conf, classes=[0], verbose=False)[0]
        for b in pres.boxes:
            persons.append(tuple(float(v) for v in b.xyxy[0]))
        res = model(frame, conf=a.conf, verbose=False)[0]
        for b in res.boxes:
            cat = category(res.names[int(b.cls)])
            if cat:
                box = tuple(float(v) for v in b.xyxy[0])
                ppe.append((cat, box))
                cv2.rectangle(frame, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:]), (200, 200, 0), 1)

        results = evaluate(persons, ppe, zone, a.require)
        overall = max((r["status"] for r in results), key=RANK.get, default="SAFE")

        if zone:
            cv2.polylines(frame, [np.array(zone, np.int32)], True, (0, 0, 255), 2)
        for r in results:
            x1, y1, x2, y2 = (int(v) for v in r["box"])
            c = COLORS[r["status"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            txt = r["status"] if not r["missing"] else f'{r["status"]}: no {", ".join(r["missing"])}'
            cv2.putText(frame, txt, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
        cv2.rectangle(frame, (0, 0), (260, 34), COLORS[overall], -1)
        cv2.putText(frame, f"STATUS: {overall}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # log only if the problem persists for a few frames (avoids flicker) and respect a cooldown
        bad_frames = bad_frames + 1 if overall != "SAFE" else 0
        if bad_frames >= a.persist and time.time() - last_log > a.cooldown:
            details = "; ".join(f'missing {",".join(r["missing"])} (in_zone={r["in_zone"]})'
                                for r in results if r["status"] != "SAFE")
            log_event(frame, overall, details)
            last_log = time.time()

        cv2.imshow(WIN, frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("f"):                             # toggle full screen
            full = not full
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()