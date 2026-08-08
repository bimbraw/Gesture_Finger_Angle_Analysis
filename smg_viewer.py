import sys
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph.opengl as gl

FILE = sys.argv[1] if len(sys.argv) > 1 else "Test_Subject_1/Slow_Speed/Fist/data.npz"
FPS = 25.0
FINGERS = ["Index", "Middle", "Ring", "Pinky"]
COLORS = [(220, 50, 50), (50, 200, 80), (60, 120, 230), (200, 80, 200)]
XBASE = [-2.4, -0.8, 0.9, 2.5]
LENGTH = [6.5, 7.6, 7.0, 5.6]
PALM = 8.0


def load(path):
    d = np.load(path)
    us = d["ultrasound"].astype(np.float32) * float(d["us_scale"])
    if "mcp_angles" in d.files:
        angles = d["mcp_angles"].astype(np.float32)
    else:
        angles = np.zeros((len(us), 4), np.float32)
    return us, angles


def preprocess(frames):
    log = 20.0 * np.log10(frames + 1e-6)
    lo, hi = np.percentile(log.ravel()[::17], [1.0, 99.5])
    return np.clip((log - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def finger_line(x_base, length, angle_deg):
    a = np.radians(angle_deg)
    direction = np.array([0.0, np.cos(a), -np.sin(a)])
    base = np.array([x_base, PALM, 0.0])
    return np.array([base, base + length * direction])


def viridis():
    pos = [0.0, 0.25, 0.5, 0.75, 1.0]
    col = [(68, 1, 84, 255), (59, 81, 139, 255), (33, 144, 140, 255),
           (92, 200, 99, 255), (253, 231, 37, 255)]
    return pg.ColorMap(pos, col).getLookupTable(0.0, 1.0, 256)


def main():
    frames, angles = load(FILE)
    frames = preprocess(frames)
    n = len(frames)
    t = np.arange(n) / FPS

    pg.setConfigOptions(imageAxisOrder="row-major")
    app = pg.mkQApp("SMG viewer")
    win = QtWidgets.QWidget()
    root = QtWidgets.QVBoxLayout(win)
    top = QtWidgets.QHBoxLayout()
    root.addLayout(top, 1)

    gv = pg.GraphicsLayoutWidget()
    vb = gv.addViewBox()
    vb.setAspectLocked(True)
    vb.invertY(True)
    img = pg.ImageItem()
    img.setLookupTable(viridis())
    vb.addItem(img)
    top.addWidget(gv, 1)

    glw = gl.GLViewWidget()
    glw.setCameraPosition(distance=30, elevation=18, azimuth=-60)
    lines = []
    for c in COLORS:
        item = gl.GLLinePlotItem(color=(c[0] / 255, c[1] / 255, c[2] / 255, 1.0), width=6, antialias=True)
        glw.addItem(item)
        lines.append(item)
    top.addWidget(glw, 1)

    plot = pg.PlotWidget()
    plot.addLegend()
    plot.setLabel("bottom", "time", "s")
    plot.setLabel("left", "MCP angle", "deg")
    for j, name in enumerate(FINGERS):
        plot.plot(t, angles[:, j], pen=pg.mkPen(COLORS[j], width=2), name=name)
    cursor = pg.InfiniteLine(angle=90, pen=pg.mkPen((230, 230, 230)))
    plot.addItem(cursor)
    root.addWidget(plot)

    controls = QtWidgets.QHBoxLayout()
    root.addLayout(controls)
    play = QtWidgets.QPushButton("Play")
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setRange(0, n - 1)
    controls.addWidget(play)
    controls.addWidget(slider, 1)

    def show(i):
        i = int(np.clip(i, 0, n - 1))
        img.setImage(frames[i], levels=(0.0, 1.0), autoLevels=False)
        cursor.setPos(t[i])
        for j in range(4):
            lines[j].setData(pos=finger_line(XBASE[j], LENGTH[j], float(angles[i, j])))

    timer = QtCore.QTimer()
    timer.setInterval(int(1000 / FPS))
    timer.timeout.connect(lambda: slider.setValue((slider.value() + 1) % n))

    def toggle():
        if timer.isActive():
            timer.stop()
            play.setText("Play")
        else:
            timer.start()
            play.setText("Pause")

    play.clicked.connect(toggle)
    slider.valueChanged.connect(show)
    show(0)
    win.resize(1100, 650)
    win.show()
    pg.exec()


if __name__ == "__main__":
    main()
