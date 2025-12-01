import socket
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter,find_peaks

# ============================================================
# 1. PARAMETERS (MODIFIED FOR 10-SECOND WINDOW)
# ============================================================
UDP_PORT = 12345
BEAGLE_IP = "192.168.7.2"

# FIX: Set FS to the actual data streaming rate (2000 Hz)
FS = 500.0                     
RAW_FS = 500.0                  

WINDOW_SEC = 10 # <-- INCREASED: Now shows 10 seconds of data
N = int(FS * WINDOW_SEC)    # N is now 20000

# Filters & peak params (now designed for 2000 Hz)
NOTCH_F0 = 60.0
NOTCH_Q = 35.0
MIN_PEAK_HEIGHT = 0.1     
MIN_DISTANCE_S = 0.25     
MIN_DISTANCE_SAMPLES = int(MIN_DISTANCE_S * FS)

# State
buffer = np.zeros(N, dtype=float)
SAMPLE_COUNTER = 0          
SAMPLE_COUNT = 0            
LAST_PEAK_TIME = 0.0
CURRENT_BPM = 0.0

# ============================================================
# SOCKET, GUI, PLOT SETUP (N is updated here)
# ============================================================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
sock.setblocking(False)
# Ask the Beagle to send (if your firmware expects it)
try:
    sock.sendto(b"send\n", (BEAGLE_IP, UDP_PORT))
except Exception:
    pass

app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Live ECG Plot @2000Hz")
win.resize(1000, 500)
win.show()

win.nextRow()
bpm_label = pg.LabelItem(justify='center')
win.addItem(bpm_label)
bpm_label.setText(f"<span style='color: white; font-size: 20pt;'>BPM: {CURRENT_BPM:.1f}</span>")

plot = win.addPlot()
plot.setLabel('left', 'Voltage', 'V')
plot.setLabel('bottom', 'Time', 's')
curve = plot.plot(pen=pg.mkPen('g', width=1.2))

# Time vector for plotting: -WINDOW_SEC .. 0
t = np.linspace(-WINDOW_SEC, 0, N) # <-- T updated with new N

# ============================================================
# FILTER DESIGN (using the new FS=2000)
# ============================================================
low_cutoff_frequency = 0.5
high_cutoff_frequency = 15.0
order = 4
nyquist_frequency = 0.5 * FS
normalized_low = low_cutoff_frequency / nyquist_frequency
normalized_high = high_cutoff_frequency / nyquist_frequency
b, a = butter(order, [normalized_low, normalized_high], btype='band')

b_notch, a_notch = iirnotch(NOTCH_F0 / nyquist_frequency, NOTCH_Q)

# ============================================================
# UPDATE LOOP (recv raw samples, filter, detect)
# ============================================================
def update():
    # Declare all global variables used/modified
    global buffer, b, a, curve, t, LAST_PEAK_TIME, CURRENT_BPM, SAMPLE_COUNT, SAMPLE_COUNTER, N, MIN_PEAK_HEIGHT, MIN_DISTANCE_SAMPLES, bpm_label
    
    newDataReceived = False

    # A) Receive as many UDP samples as available
    try:
        while True:
            data, addr = sock.recvfrom(1024)
            text = data.decode().strip()
            try:
                raw_v = float(text)    # expect C to send the voltage (e.g., 0.0..3.3)
            except ValueError:
                continue

            SAMPLE_COUNT += 1
            # push into circular buffer: oldest at index 0, newest at -1
            buffer = np.roll(buffer, -1)
            buffer[-1] = raw_v

            SAMPLE_COUNTER += 1
            newDataReceived = True

    except BlockingIOError:
        # no more UDP data now
        pass

    # B) Nothing new -> don't process
    if not newDataReceived:
        return

    # Remove DC offset (on the buffer)
    voltage_dc_removed = buffer - np.mean(buffer)
    voltage_filtered = voltage_dc_removed.copy()

    # Only apply filtfilt after we have a full buffer (Stabilization)
    if SAMPLE_COUNTER >= N:
        try:
            x_filtered_bp = filtfilt(b, a, voltage_dc_removed)
            voltage_filtered = filtfilt(b_notch, a_notch, x_filtered_bp)
        except Exception as e:
            # filtfilt can fail if something odd happens; keep DC-removed signal
            # print("Filtering error:", e) # Uncomment for debugging filter instability
            voltage_filtered = voltage_dc_removed.copy()

        # Peak detection on filtered signal
        peaks, properties = find_peaks(
            voltage_filtered,
            height=MIN_PEAK_HEIGHT,
            distance=MIN_DISTANCE_SAMPLES
        )

        if len(peaks) > 0:
            last_peak_index = peaks[-1]  # index into buffer [0..N-1]

            # convert buffer index to absolute sample index:
            absolute_index = SAMPLE_COUNT - (N - last_peak_index)
            time_of_latest_peak = absolute_index / float(FS)  # seconds since program start

            # accept only if newer than last
            if time_of_latest_peak > LAST_PEAK_TIME:
                if LAST_PEAK_TIME == 0.0:
                    # seed baseline time
                    LAST_PEAK_TIME = time_of_latest_peak
                else:
                    rr_interval_s = time_of_latest_peak - LAST_PEAK_TIME
                    # sanity check for RR interval (tunable)
                    if 0.12 < rr_interval_s < 3.0:    # allow up to 500 BPM lower bound ~0.12s
                        CURRENT_BPM = 60.0 / rr_interval_s
                        LAST_PEAK_TIME = time_of_latest_peak
                        bpm_label.setText(
                            f"<span style='color: #FF69B4; font-size: 30pt; font-weight: bold;'>BPM: {CURRENT_BPM:.1f}</span>"
                        )
                    else:
                        # unrealistic interval, update to avoid stuck comparisons
                        LAST_PEAK_TIME = time_of_latest_peak

    # Update plot (plot the filtered signal if available)
    curve.setData(t, voltage_filtered)

# ============================================================
# TIMER START
# ============================================================
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(5)    # poll frequently to collect UDP packets

print(f"Listening for ECG data on UDP port {UDP_PORT}. Processing at {FS} Hz.")
app.exec_()