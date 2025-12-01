import socket
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter
import csv

# ============================================================
#               USER PARAMETERS
# ============================================================
UDP_PORT = 12345
FS = 500              # sampling rate from Beagle
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

# Smoothing + "certain point" thresholds
SAVGOL_WINDOW = 21     # must be odd; ~42 ms at 500 Hz
SAVGOL_ORDER  = 3
JUMP_THRESH   = 150.0  # mV, max allowed change between samples
AMP_THRESH    = 350.0  # mV, max deviation from median to keep

# CSV logging
LOG_FILENAME = "ecg_log.csv"

# ============================================================
#                  1. SET UP UDP SOCKET
# ============================================================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
sock.setblocking(False)
sock.sendto(b"send\n", (BEAGLE_IP, UDP_PORT))

# ============================================================
#                  1a. SET UP CSV LOGGING
# ============================================================

csv_file = open(LOG_FILENAME, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["sample_index", "time_s", "voltage_mV"])

sample_index = 0  # global sample counter

# ============================================================
#                  2. SET UP DATA BUFFER
# ============================================================

buffer = np.zeros(N, dtype=float)

print(">>> Running ECG plot script with filters + certainty masking (tuned)")
print(f">>> Logging raw samples to {LOG_FILENAME}")

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
plot.setLabel('left', 'Voltage', 'mV')
plot.setLabel('bottom', 'Time', 's')

curve = plot.plot(
    pen=None,          # no connecting line
    symbol='o',        # point marker
    symbolSize=4,
    symbolBrush='g'
)

t = np.linspace(-WINDOW_SEC, 0, N)

# ============================================================
#                  4. UPDATE FUNCTION (TIMER)
# ============================================================

def update():
    global buffer, curve, t, sample_index

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

            # ---- LOG TO CSV (raw value) ----
            time_s = sample_index / FS
            csv_writer.writerow([sample_index, time_s, v])
            sample_index += 1

            # Occasionally flush to disk
            if sample_index % 100 == 0:
                csv_file.flush()

            # ---- Update circular buffer for plotting ----
            buffer = np.roll(buffer, -1)
            buffer[-1] = v

    except BlockingIOError:
        pass

    # --------------------------------------------------------
    # DIGITAL FILTERING + CERTAINTY MASK
    # --------------------------------------------------------
    x = buffer - np.mean(buffer)  # remove DC

    if np.any(x):
        # 1) band-pass 0.5–25 Hz
        x_filt = filtfilt(b_bp, a_bp, x)

        # 2) 60 Hz notch
        x_filt = filtfilt(b_notch, a_notch, x_filt)

        # 3) smoothing (Savitzky–Golay)
        win_len = min(SAVGOL_WINDOW, len(x_filt) - (1 - len(x_filt) % 2))
        if win_len < 3:
            x_smooth = x_filt
        else:
            if win_len % 2 == 0:
                win_len -= 1
            x_smooth = savgol_filter(
                x_filt,
                window_length=win_len,
                polyorder=min(SAVGOL_ORDER, win_len - 1)
            )

        # 4) "certain point" mask
        # 4a) reject big jumps between consecutive samples
        diff = np.abs(np.diff(x_smooth))
        jump_mask = np.concatenate(([True], diff < JUMP_THRESH))

        # 4b) reject amplitude outliers vs median
        med = np.median(x_smooth)
        amp_mask = np.abs(x_smooth - med) < AMP_THRESH

        mask = jump_mask & amp_mask

        t_clean = t[mask]
        y_clean = x_smooth[mask]
    else:
        t_clean = t
        y_clean = x

    # --------------------------------------------------------
    # Update plot with CLEAN points only
    # --------------------------------------------------------
    curve.setData(t_clean, y_clean)

# ============================================================
#           5. SET UP QT TIMER TO CALL update()
# ============================================================

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)          # update every 10 ms

# ============================================================
#           6. CLEANUP ON EXIT
# ============================================================

def on_exit():
    try:
        sock.close()
    except Exception:
        pass
    try:
        csv_file.flush()
        csv_file.close()
    except Exception:
        pass

app.aboutToQuit.connect(on_exit)

# ============================================================
#           7. START APPLICATION EVENT LOOP
# ============================================================

print("Ready to receive ECG data over UDP port", UDP_PORT)
app.exec_()
