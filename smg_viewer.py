import sys
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import pyqtgraph.opengl as gl

FILE = sys.argv[1] if len(sys.argv) > 1 else "Test_Subject_1/Slow_Speed/Fist/data.npz"
FPS = 25.0
PCT_LO, PCT_HI = 1.0, 99.5
JOINT_RATIOS = (1.0, 0.9, 0.6)
FINGERS = ["Index", "Middle", "Ring", "Pinky"]
PLOT_COLORS = {"Index": (220, 50, 50), "Middle": (50, 200, 80),
               "Ring": (60, 120, 230), "Pinky": (200, 80, 200)}

PALM_LEN = 8.0
PALM_THICK = 1.4
FINGER_GEOM = {"Index": (-2.4, 6.5), "Middle": (-0.8, 7.6),
               "Ring": (0.9, 7.0), "Pinky": (2.5, 5.6)}
PHALANX_FRAC = (0.45, 0.30, 0.25)
PHALANX_RADIUS = (0.50, 0.42, 0.34)
JOINT_RADIUS = 0.52
THUMB_RADIUS = 0.58
METAL = (0.72, 0.74, 0.80, 1.0)
JOINT_COLOR = (0.40, 0.42, 0.47, 1.0)
PALM_COLOR = (0.30, 0.32, 0.37, 1.0)
_TX = min(g[0] for g in FINGER_GEOM.values()) - 0.6
THUMB_JOINTS = [(_TX + 1.0, 3.0, 0.0), (_TX + 0.3, 3.6, -1.7), (_TX + 0.5, 4.2, -3.2)]


def load(path):
    d = np.load(path)
    us = d["ultrasound"].astype(np.float32) * float(d["us_scale"])
    if "mcp_angles" in d.files:
        angles = d["mcp_angles"].astype(np.float32)
    else:
        angles = np.zeros((len(us), 4), np.float32)
    return us, angles


def preprocess(frames):
    log = 20.0 * np.log10(np.asarray(frames, np.float32) + 1e-6)
    lo, hi = np.percentile(log.ravel()[::17], [PCT_LO, PCT_HI])
    return np.clip((log - lo) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)


def finger_joints(x_base, total_len, theta_deg, ratios=JOINT_RATIOS):
    mcp = np.array([x_base, PALM_LEN, 0.0])
    pts, p, cum = [mcp], mcp.copy(), 0.0
    for k, frac in enumerate(PHALANX_FRAC):
        cum += theta_deg * ratios[k]
        th = np.radians(cum)
        d = np.array([0.0, np.cos(th), -np.sin(th)])
        p = p + frac * total_len * d
        pts.append(p.copy())
    return np.array(pts)


def perp_basis(d):
    d = np.asarray(d, float)
    nn = np.linalg.norm(d)
    dh = d / nn if nn > 0 else np.array([0.0, 0.0, 1.0])
    ref = np.array([0.0, 0.0, 1.0]) if abs(dh[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    b1 = np.cross(ref, dh); b1 /= np.linalg.norm(b1)
    b2 = np.cross(dh, b1)
    return b1, b2, dh


def segment_matrix(p0, p1, radius):
    p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
    length = np.linalg.norm(p1 - p0)
    b1, b2, dh = perp_basis(p1 - p0)
    m = np.eye(4)
    m[:3, 0] = b1 * radius
    m[:3, 1] = b2 * radius
    m[:3, 2] = dh * length
    m[:3, 3] = p0
    return m


def sphere_matrix(center, radius):
    m = np.eye(4)
    m[0, 0] = m[1, 1] = m[2, 2] = radius
    m[:3, 3] = np.asarray(center, float)
    return m


def box_mesh(center, size):
    cx, cy, cz = center
    sx, sy, sz = size
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    f = np.array([[0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6], [0, 4, 5], [0, 5, 1],
                  [1, 5, 6], [1, 6, 2], [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]])
    return v, f


def viridis_lut():
    a = [(0.0, (68, 1, 84)), (0.13, (71, 44, 122)), (0.25, (59, 81, 139)),
         (0.38, (44, 113, 142)), (0.5, (33, 144, 140)), (0.63, (39, 173, 129)),
         (0.75, (92, 200, 99)), (0.88, (170, 220, 50)), (1.0, (253, 231, 37))]
    cm = pg.ColorMap([p for p, _ in a], [(c[0], c[1], c[2], 255) for _, c in a])
    return cm.getLookupTable(0.0, 1.0, 256)


def main():
    us, angles = load(FILE)
    frames = preprocess(us)
    n = min(len(frames), len(angles))
    frames = frames[:n]
    angles = np.nan_to_num(angles[:n], nan=0.0).astype(np.float32)
    t = np.arange(n) / FPS

    pg.setConfigOptions(imageAxisOrder="row-major", antialias=False)
    app = pg.mkQApp("SMG viewer")
    win = QtWidgets.QWidget()
    win.setWindowTitle("SMG viewer")
    rootlyt = QtWidgets.QVBoxLayout(win)
    hsplit = QtWidgets.QSplitter(QtCore.Qt.Horizontal); rootlyt.addWidget(hsplit, 1)

    gv = pg.GraphicsLayoutWidget()
    vb = gv.addViewBox(); vb.setAspectLocked(True); vb.invertY(True)
    img = pg.ImageItem(); img.setLookupTable(viridis_lut()); vb.addItem(img)
    gv.setMinimumSize(220, 260)
    hsplit.addWidget(gv)

    rsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical); hsplit.addWidget(rsplit)

    glw = gl.GLViewWidget(); glw.setMinimumSize(220, 180)
    glw.setCameraPosition(pos=pg.Vector(0.0, 7.0, 0.5), distance=34, elevation=18, azimuth=-60)
    rsplit.addWidget(glw)

    def qmat(m):
        return QtGui.QMatrix4x4(*[float(x) for x in m.flatten()])

    cyl = gl.MeshData.cylinder(rows=1, cols=12, radius=[1.0, 1.0], length=1.0)
    sph = gl.MeshData.sphere(rows=8, cols=12, radius=1.0)

    def add_cyl(color):
        it = gl.GLMeshItem(meshdata=cyl, color=color, smooth=True, shader="shaded"); glw.addItem(it); return it

    def add_sph(color):
        it = gl.GLMeshItem(meshdata=sph, color=color, smooth=True, shader="shaded"); glw.addItem(it); return it

    xs = [FINGER_GEOM[f][0] for f in FINGERS]
    pv, pf = box_mesh((0.0, PALM_LEN / 2, 0.0), ((max(xs) - min(xs)) + 2.0, PALM_LEN, PALM_THICK))
    glw.addItem(gl.GLMeshItem(vertexes=pv, faces=pf, color=PALM_COLOR, smooth=False, shader="shaded"))

    for a, b in zip(THUMB_JOINTS[:-1], THUMB_JOINTS[1:]):
        add_cyl(METAL).setTransform(qmat(segment_matrix(a, b, THUMB_RADIUS)))
    for j in THUMB_JOINTS:
        add_sph(JOINT_COLOR).setTransform(qmat(sphere_matrix(j, THUMB_RADIUS)))

    seg_items = {f: [add_cyl(METAL) for _ in range(3)] for f in FINGERS}
    jnt_items = {f: [add_sph(JOINT_COLOR) for _ in range(4)] for f in FINGERS}

    plot = pg.PlotWidget(); plot.setMinimumHeight(150)
    plot.addLegend(offset=(-10, 8))
    plot.setLabel("bottom", "time", "s")
    plot.setLabel("left", "MCP angle", "deg")
    for j, f in enumerate(FINGERS):
        plot.plot(t, angles[:, j], pen=pg.mkPen(PLOT_COLORS[f], width=2), name=f)
    vline = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen((235, 235, 235), width=1))
    plot.addItem(vline)
    rsplit.addWidget(plot)
    rsplit.setSizes([460, 240])
    hsplit.setSizes([640, 560])

    lbl = QtWidgets.QLabel()
    lbl.setStyleSheet("font-family: Consolas, monospace; font-size: 15px;")
    rootlyt.addWidget(lbl)

    ctl = QtWidgets.QHBoxLayout(); rootlyt.addLayout(ctl)
    play_btn = QtWidgets.QPushButton("Play")
    frame_sld = QtWidgets.QSlider(QtCore.Qt.Horizontal); frame_sld.setRange(0, n - 1)
    fps_box = QtWidgets.QSpinBox(); fps_box.setRange(1, 120)
    fps_box.setValue(int(FPS)); fps_box.setSuffix(" fps")
    ctl.addWidget(play_btn)
    ctl.addWidget(QtWidgets.QLabel("Frame")); ctl.addWidget(frame_sld, 1)
    ctl.addWidget(QtWidgets.QLabel("Rate")); ctl.addWidget(fps_box)

    def show_frame(i):
        i = int(np.clip(i, 0, n - 1))
        img.setImage(frames[i], levels=(0.0, 1.0), autoLevels=False)
        vline.setPos(t[i])
        for j, f in enumerate(FINGERS):
            p = finger_joints(FINGER_GEOM[f][0], FINGER_GEOM[f][1], float(angles[i, j]))
            for k in range(3):
                seg_items[f][k].setTransform(qmat(segment_matrix(p[k], p[k + 1], PHALANX_RADIUS[k])))
            for k in range(4):
                jnt_items[f][k].setTransform(qmat(sphere_matrix(p[k], JOINT_RADIUS)))
        lbl.setText("Frame %4d/%d   %d fps      " % (i + 1, n, fps_box.value())
                    + "   ".join("%-6s %5.1f°" % (f, angles[i, j])
                                 for j, f in enumerate(FINGERS)))

    timer = QtCore.QTimer()
    try:
        timer.setTimerType(QtCore.Qt.PreciseTimer)
    except Exception:
        pass
    timer.setInterval(int(1000 / FPS))
    play_state = {"t0": 0.0, "f0": 0}

    def tick():
        fps = fps_box.value()
        target = (play_state["f0"] + int((time.perf_counter() - play_state["t0"]) * fps)) % n
        if target != frame_sld.value():
            frame_sld.setValue(target)

    def on_fps(v):
        timer.setInterval(max(1, int(1000 / v)))
        play_state.update(t0=time.perf_counter(), f0=frame_sld.value())

    def toggle():
        if timer.isActive():
            timer.stop(); play_btn.setText("Play")
        else:
            play_state.update(t0=time.perf_counter(), f0=frame_sld.value())
            timer.start(); play_btn.setText("Pause")

    timer.timeout.connect(tick)
    fps_box.valueChanged.connect(on_fps)
    play_btn.clicked.connect(toggle)
    frame_sld.valueChanged.connect(show_frame)

    show_frame(0)
    vb.autoRange()
    win.resize(1220, 760)
    win.show()
    pg.exec()


if __name__ == "__main__":
    main()
