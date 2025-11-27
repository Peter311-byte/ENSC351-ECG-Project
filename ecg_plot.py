import socket
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter
import csv
import time

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
LP_CUTOFF = 25.0       # low-pass cutoff (Hz)
BP_ORDER  = 4

# 60 Hz notch
NOTCH_F0  = 60.0       # notch frequency (Hz)
NOTCH_Q   = 35.0       # quality factor

# Savitzky–Golay smoothing
SAVGOL_WINDOW = 11     # must be odd
SAVGOL_ORDER  = 3

# ============================================================
#                  SET UP CSV LOGGING
# ============================================================

csv_file = open("ecg_recording.csv", "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp_sec", "voltage_V"])

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
print(">>> Recording data to ecg_recording.csv")

# ============================================================
#                  2a. DESIGN FILTERS
# ============================================================

nyq = 0.5 * FS

# Band-pass Butterworth
low = HP_CUTOFF / nyq
high = LP_CUTOFF / nyq
b_bp, a_bp = butter(BP_ORDER, [low, high], btype='bandpass')

# 60 Hz notch
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

    try:
        while True:
            data, addr = sock.recvfrom(1024)
            text = data.decode().strip()

            try:
                v = float(text)
            except ValueError:
                continue

            # Write to CSV
            csv_writer.writerow([time.time(), v])
            csv_file.flush()

            # Update buffer
            buffer = np.roll(buffer, -1)
            buffer[-1] = v

    except BlockingIOError:
        pass

    # --------------------------------------------------------
    # FILTERING PIPELINE
    # --------------------------------------------------------

    x = buffer - np.mean(buffer)

    if np.any(x):
        # Band-pass
        x_filt = filtfilt(b_bp, a_bp, x)

        # Notch
        x_filt = filtfilt(b_notch, a_notch, x_filt)

        # Smoothing
        win_len = min(SAVGOL_WINDOW, len(x_filt) - (1 - len(x_filt) % 2))
        if win_len < 3:
            x_smooth = x_filt
        else:
            if win_len % 2 == 0:
                win_len -= 1
            x_smooth = savgol_filter(x_filt, window_length=win_len,
                                     polyorder=min(SAVGOL_ORDER, win_len - 1))
    else:
        x_smooth = x

    # Update plot
    curve.setData(t, x_smooth)

# ============================================================
#           5. START QT TIMER
# ============================================================

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)     # update every 10 ms

# ============================================================
#           6. START APPLICATION
# ============================================================

print("Ready to receive ECG data over UDP port", UDP_PORT)
app.exec_()
