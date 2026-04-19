ClearVisionAI

ClearVisionAI is an AI-based system designed to reduce the risk of visually triggered epileptic seizures by detecting rapid and potentially harmful screen changes in real time.

Disclaimer

This project is not a medical device and does not guarantee prevention of epileptic seizures. It is intended as an experimental safety layer to reduce visual risk factors.

Features
Real-time screen analysis
Detection of high-frequency visual changes (flickering, flashes)
AI-based risk estimation (RiskNet)
Safety filtering system (SafetyEngine)
Risk smoothing to avoid unstable output
Overlay system for live feedback
How It Works
Screen Capture
The program continuously captures the screen using MSS.
Feature Extraction
Visual features are extracted from each frame (brightness changes, motion, flicker intensity).
Risk Evaluation
A neural network (RiskNet) estimates how dangerous the current frame sequence is.
Physics-Based Validation
Additional checks (e.g. flicker frequency) are applied using physical rules.
Safety Engine
The SafetyEngine stabilizes and filters the output.
Overlay Output
A real-time risk indicator is displayed on screen.
Installation
git clone https://github.com/YOUR_USERNAME/ClearVisionAI.git
cd ClearVisionAI
pip install -r requirements.txt
Usage
python main.py

The system will start analyzing your screen and display a live risk indicator.

Project Structure
.
├── main.py
├── model/
│   └── net.py
├── vision/
│   ├── features.py
│   ├── physics.py
│   └── safety_engine.py
└── requirements.txt
Requirements
Python 3.x
OpenCV
NumPy
MSS
Future Improvements
Reduce false positives
Improve AI training dataset
GPU optimization
More accurate flicker detection (Hz-based)
Full-screen adaptive dimming / safe rendering
Contributing

Contributions are currently not open. Please contact the author for permission.

License
Copyright (c) 2026 Pro_cessor6

All rights reserved.

This project and its source code may not be copied, modified, distributed, or used without explicit permission from the author.
