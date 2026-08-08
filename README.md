# Gesture_Finger_Angle_Analysis

Reference code to visualize the forearm ultrasound and finger-angle dataset. It
plays one recording session with the ultrasound video (viridis, log-scaled) on
the left, and on the right a 3D robotic hand above the four MCP joint-angle
traces, all driven by the recorded angles, with Play, Frame, and Rate controls.

## Dataset

Download from Zenodo: https://doi.org/10.5281/zenodo.21825018

Each subject is one ZIP archive that extracts to
`Test_Subject_N/<Speed>_Speed/<Gesture>/data.npz`, where `<Speed>` is `Slow`,
`Medium`, or `Fast`. Each `data.npz` contains:

- `ultrasound` — `(N, 636, 256)` float16; recover the envelope with `ultrasound.astype("float32") * us_scale`
- `us_scale` — float32
- `mcp_angles` — `(N, 4)` float32, MCP angles in degrees for [Index, Middle, Ring, Pinky]

## Install

```
pip install -r requirements.txt
```

## Run

1. Download a subject archive (for example `Test_Subject_1.zip`) from the Zenodo
   record and extract it.
2. Point the viewer at any `data.npz` inside the extracted folder:

```
python smg_viewer.py Test_Subject_1/Slow_Speed/Fist/data.npz
```

With no argument it defaults to `Test_Subject_1/Slow_Speed/Fist/data.npz`
relative to the current folder.

## Training models

To train hand-gesture models on forearm ultrasound, use
[bimbraw/Acoustic_Gesture_Recognition](https://github.com/bimbraw/Acoustic_Gesture_Recognition)
as a reference.

## Papers

- K. Bimbraw et al., "Simultaneous Estimation of Hand Configurations and Finger Joint Angles Using Forearm Ultrasound," IEEE Transactions on Medical Robotics and Bionics, 2023. https://doi.org/10.1109/TMRB.2023.3237774
- K. Bimbraw et al., "Prediction of Metacarpophalangeal Joint Angles and Classification of Hand Configurations Based on Ultrasound Imaging of the Forearm," IEEE ICRA, 2022. https://doi.org/10.1109/ICRA46639.2022.9812287

## License

Data: CC BY 4.0. Code: MIT.
