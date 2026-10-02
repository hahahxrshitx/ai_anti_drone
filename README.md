# AI-Powered Anti-Drone Surveillance & Threat Detection System

An AI-based surveillance prototype for detecting drones, tracking their movement, analysing possible threats, and predicting their future trajectory.

---

## Overview

Drones are becoming increasingly common, but simply detecting a drone is not always enough. A surveillance system also needs to understand where the drone is moving and whether its behaviour could indicate a potential threat.

For this project, we built a prototype that combines computer vision with simulated sensor data to detect and analyse drone activity.

The system takes video input, detects drones using YOLO, tracks their movement, combines the visual information with simulated RF and GPS data, evaluates the situation, and predicts the drone's trajectory.

Since we did not have access to physical RF and GPS hardware, these inputs were simulated during development and testing.

---

## What the System Does

* Detects drones from video input
* Tracks detected drones across frames
* Generates simulated RF and GPS sensor data
* Analyses drone movement and behaviour
* Identifies activity around restricted areas
* Assigns threat levels based on the available information
* Predicts future movement using a Kalman Filter
* Displays detections, trajectories and alerts through the monitoring interface

---

## System Workflow

```text
Video Input
     ↓
YOLO Drone Detection
     ↓
Drone Tracking
     ↓
Simulated RF + GPS Data
     ↓
Threat Assessment
     ↓
Kalman Filter
     ↓
Trajectory Prediction
     ↓
Alerts / Monitoring Dashboard
```

---

## How AI Is Used

### 1. Drone Detection

YOLO is used to identify drones in surveillance video and return their locations using bounding boxes and confidence scores.

### 2. Tracking

After detection, the system tracks drone movement across frames so that the movement of an individual drone can be monitored over time.

### 3. Threat Assessment

The system combines information such as drone location, movement behaviour, restricted-area activity and simulated sensor information to assess potential threats.

Threat levels are represented as:

* **Low**
* **Medium**
* **High**

### 4. Trajectory Prediction

A Kalman Filter is used to estimate the drone's future position from its previous movement. This helps the system identify possible changes in its trajectory and potential movement towards restricted areas.

### 5. Sensor Simulation

Because physical RF and GPS sensors were not available, we created simulated sensor inputs to test how multiple sources of information could be combined with visual detection.

---

## Tech Stack

**Language**

* Python

**AI / Computer Vision**

* YOLO
* OpenCV
* Computer Vision

**Data Processing**

* NumPy
* Pandas

**Tracking & Prediction**

* Kalman Filter

**Other**

* Simulated RF data
* Simulated GPS / telemetry data
* Visualization and dashboard components

---

## Testing

We tested the system at different levels to check both individual components and the complete workflow.

Testing included:

* Unit testing
* Integration testing
* Functional testing
* Performance testing
* Simulation-based testing
* Accuracy validation

We tested scenarios involving drone detection, tracking, threat assessment, alerts and simulated sensor inputs. The project was also evaluated under different visual conditions such as changes in brightness, blur and noise.

---

## Results

In our tested simulation environment, the system achieved the following reported results:

| Metric                     |       Result |
| -------------------------- | -----------: |
| Accuracy                   |        98.5% |
| Precision                  |        97.8% |
| Recall                     |        98.2% |
| F1 Score                   |        98.0% |
| Processing Speed           |    25–30 FPS |
| Detection-to-alert latency | ~0.2 seconds |

These results are from the project's simulation and testing environment and should not be interpreted as real-world deployment performance.

---

## Limitations

This was a simulation-based college project, so there are several limitations:

* RF and GPS inputs were simulated rather than collected from physical sensors.
* The system was not tested with real drones or real RF signals.
* Detection performance can be affected by lighting, weather and camera quality.
* Different drone types may require additional training data.
* Performance can change when multiple drones are present at the same time.

---

## Future Improvements

Some improvements we would explore in a future version include:

* Testing with real RF, GPS and camera hardware
* Adding more diverse drone datasets
* Improving trajectory prediction
* Supporting additional sensor types such as radar or LiDAR
* Optimizing the system for edge devices
* Improving the monitoring dashboard
* Testing the system in real-world environments

---


## What I Learned

This project helped me understand that building an AI system is more than just training or using a model.

We had to think about the complete flow from video input and detection to tracking, sensor data, decision-making and visualization. Working with simulated sensor data also helped us understand how different sources of information can be combined when building a larger AI system.

---

## Project

GitHub:
https://github.com/hahahxrshitx/ai_anti_drone
