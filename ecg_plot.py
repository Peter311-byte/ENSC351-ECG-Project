import socket
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter

# ============================================================
#               USER PARAMETERS
# ============================================================
UDP_PORT = 12345
FS = 500               # sampling rate from Beagle
WINDOW_SEC = 5         # how many seconds to display
N = FS * WINDOW_SEC    # total buffer size

BEAGLE_IP = "192.168.7.2"

# -------- Filter params -------------------------------------
# Band-pass for ECG
HP_CUTOFF = 0.5        # high-pass cutoff (Hz)
LP_CUTOFF = 25.0       # low-pass cutoff (Hz)  (tighter to reduce EMG)
BP_ORDER  = 4

# 60 Hz notch
NOTCH_F0  = 60.0       # notch frequency (Hz)
NOTCH_Q   = 35.0       # quality factor (higher = narrower notch)

# Savitzky–Golay smoothing
SAVGOL_WINDOW = 11     # must be odd; 11 samples ≈ 22 ms at 500 Hz
SAVGOL_ORDER  = 3

# ============================================================
#                  1. SET UP UDP SOCKET
# ============================================================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
sock.setblocking(False)
sock.sendto(b"send\n", (BEAGLE_IP, UDP_PORT))

# ============================================================
#                  2. SET UP DATA BUFFER
# ============================================================

buffer = np.zeros(N, dtype=float)

print(">>> Running ECG plot script with band-pass + 60 Hz notch + smoothing")

# ============================================================
#                  2a. DESIGN FILTERS
# ============================================================

nyq = 0.5 * FS

# Band-pass Butterworth
low = HP_CUTOFF / nyq
high = LP_CUTOFF / nyq
b_bp, a_bp = butter(BP_ORDER, [low, high], btype='bandpass')

# 60 Hz notch (IIR)
b_notch, a_notch = iirnotch(NOTCH_F0 / nyq, NOTCH_Q)

# ============================================================
#                  3. SET UP PLOT WINDOW
# ============================================================

app = QtWidgets.QApplication([])

win = pg.GraphicsLayoutWidget(title="Live ECG Plot")
win.resize(800, 400)
win.show()

plot = win.addPlot()
plot.setLabel('left', 'Voltage', 'V')
plot.setLabel('bottom', 'Time', 's')

curve = plot.plot(pen=pg.mkPen(color='g', width=2))


t = np.linspace(-WINDOW_SEC, 0, N)

# ============================================================
#                  4. UPDATE FUNCTION (TIMER)
# ============================================================

def update():
    global buffer, curve, t

    # --------------------------------------------------------
    # Read all available UDP packets (non-blocking)
    # --------------------------------------------------------
    try:
        while True:
            data, addr = sock.recvfrom(1024)
            text = data.decode().strip()

            try:
                v = float(text)
                # print("got:", v)   # uncomment for debugging
            except ValueError:
                continue

            buffer = np.roll(buffer, -1)
            buffer[-1] = v

    except BlockingIOError:
        pass

    # --------------------------------------------------------
    # DIGITAL FILTERING
    # --------------------------------------------------------
    # 1) remove DC offset
    x = buffer - np.mean(buffer)

    if np.any(x):  # avoid filtering all zeros at startup
        # 2) band-pass 0.5–25 Hz
        x_filt = filtfilt(b_bp, a_bp, x)

        # 3) 60 Hz notch
        x_filt = filtfilt(b_notch, a_notch, x_filt)

        # 4) light smoothing
        # ensure window length is not larger than the signal
        win_len = min(SAVGOL_WINDOW, len(x_filt) - (1 - len(x_filt) % 2))
        if win_len < 3:   # fallback if buffer is too short
            x_smooth = x_filt
        else:
            if win_len % 2 == 0:
                win_len -= 1
            x_smooth = savgol_filter(x_filt, window_length=win_len,
                                     polyorder=min(SAVGOL_ORDER, win_len - 1))
    else:
        x_smooth = x

    # --------------------------------------------------------
    # Update plot with FILTERED data
    # --------------------------------------------------------
    curve.setData(t, x_smooth)

# ============================================================
#           5. SET UP QT TIMER TO CALL update()
# ============================================================

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(1000)          # update every 10 ms

# ============================================================
#           6. START APPLICATION EVENT LOOP
# ============================================================

print("Ready to receive ECG data over UDP port", UDP_PORT)
app.exec_()
