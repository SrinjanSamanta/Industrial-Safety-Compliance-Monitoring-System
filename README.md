# Industrial-Safety-Compliance-Monitoring-System
### Project Overview

Real-time computer vision system for industrial safety compliance. Uses YOLOv8 for person detection, foot-plane spatial geofencing for restricted machinery zones, aspect-ratio fall detection for man-down alerts, and an SQLite-backed Streamlit dashboard.mmand center.

### Core Architectural Components

* **YOLOv8 Detection Engine**: Powered by Ultralytics YOLOv8, capable of running standard person detection or custom-trained weights (such as `ppe.pt`) to monitor worker presence and equipment compliance.
* **Foot-Plane Spatial Geofencing**: Utilizes OpenCV polygon containment testing (`cv2.pointPolygonTest`) mapped specifically to worker foot coordinates. This design eliminates false positives caused by 2D camera perspective distortion, ensuring accurate boundary enforcement around restricted machinery zones.
* **Aspect-Ratio Fall Detection**:
Monitors each worker’s bounding-box width-to-height ratio in real time. A ratio above 1.2 flags a possible slip, trip, or man-down event, and a time-based check (default 5 s) confirms it before raising an alert
* **Event Throttling & SQLite Logging**: Automatically captures safety breaches, logs photographic evidence paths, and timestamps incidents into a local SQLite database with built-in cooldown logic to prevent log spamming.
* **Streamlit Command Center**: Features a dynamic, web-based dashboard that provides safety managers with real-time compliance metrics, historical event logs, and operational telemetry.
