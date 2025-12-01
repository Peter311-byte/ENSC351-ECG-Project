import socket
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore
from scipy.signal import butter, filtfilt, iirnotch, savgol_filter,find_peaks

# ============================================================
# 1. PARAMETERS
# ============================================================
UDP_PORT = 12345
FS = 500
WINDOW_SEC = 5
N = FS*WINDOW_SEC
BEAGLE_IP = "192.168.7.2"

# --- Filter and Peak Params ---
NOTCH_F0 = 60.0    # Notch frequency (Hz)
NOTCH_Q = 35.0     # Quality factor
MIN_PEAK_HEIGHT = 0.6 # Height threshold for R-peak detection
MIN_DISTANCE_S = 0.25 # Min time between peaks (for max 240 BPM)
MIN_DISTANCE_SAMPLES = int(MIN_DISTANCE_S * FS)

# --- Heart Rate Variables (Persistent State) ---
LAST_PEAK_TIME = 0.0
CURRENT_BPM = 0.0
PEAK_HISTORY = np.zeros(10) # Used for potential averaging (not fully implemented here)


# ============================================================
# 2. SETUP: Socket, Buffer, and GUI
# ============================================================
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))
sock.setblocking(False)
sock.sendto(b"send\n", (BEAGLE_IP, UDP_PORT))

buffer = np.zeros(N,dtype=float)
app = QtWidgets.QApplication([])

# Setup Plot Window with two rows: BPM label and ECG plot
win = pg.GraphicsLayoutWidget(title="Live ECG Plot")
win.resize(800, 400)
win.show()

# Row 1: BPM Display Label
win.nextRow()
bpm_label = pg.LabelItem(justify='center')
win.addItem(bpm_label)
bpm_label.setText(f"<span style='color: white; font-size: 20pt;'>BPM: {CURRENT_BPM:.1f}</span>")

# Row 2: ECG Plot
plot = win.addPlot()
plot.setLabel('left', 'Voltage', 'mV')
plot.setLabel('bottom', 'Time', 's')

curve = plot.plot(
    pen = pg.mkPen('g',width = 1.5),
    symbol = None,
)

t = np.linspace(-WINDOW_SEC, 0, N)


# ============================================================
# 3. FILTER DESIGN
# ============================================================
low_cutoff_frequency = 0.5 
high_cutoff_frequency = 15.0 
order = 4
nyquist_frequency = 0.5 * FS

normalized_low = low_cutoff_frequency / nyquist_frequency
normalized_high = high_cutoff_frequency / nyquist_frequency

# Bandpass Filter Coefficients
b, a = butter(order, [normalized_low, normalized_high], btype='band') 

# Notch Filter Coefficients
b_notch, a_notch = iirnotch(NOTCH_F0 / nyquist_frequency, NOTCH_Q)


# ============================================================
# 4. UPDATE FUNCTION (ECG and BPM Calculation)
# ============================================================

def update():
    # FIXED: Added bpm_label to global list as it is modified
    global buffer, b, a, curve, t, order, b_notch, a_notch, LAST_PEAK_TIME, CURRENT_BPM, PEAK_HISTORY, MIN_PEAK_HEIGHT, MIN_DISTANCE_SAMPLES, bpm_label
    
    newDataRecieved = False 
    # current_time_s is 0.0, representing the latest point in the buffer
    current_time_s = t[-1] 
    
    # ------------------------------------------------------------------
    # A. RECEIVE DATA AND UPDATE BUFFER
    # ------------------------------------------------------------------
    try:
        while True:
            data, addr = sock.recvfrom(1024)
            text = data.decode().strip()
            
            try:
                voltage = float(text)
            except ValueError:
                continue

            buffer = np.roll(buffer,-1)
            buffer[-1] = voltage
            newDataRecieved = True

    except BlockingIOError:
        pass
    
    # ------------------------------------------------------------------
    # B. PROCESS, CALCULATE, AND PLOT
    # ------------------------------------------------------------------
    if newDataRecieved:
        
        # 1. Remove DC offset
        voltage_dc_removed = buffer - np.mean(buffer)

        # 2. Filtering Pipeline
        if len(buffer) > 2 * order + 1:
            # Bandpass Filter
            x_filtered_bp = filtfilt(b, a, voltage_dc_removed)
            # Notch Filter
            voltage_filtered = filtfilt(b_notch, a_notch, x_filtered_bp)
        else:
            voltage_filtered = voltage_dc_removed 
            
        # 3. Peak Detection for BPM Calculation
        peaks, properties = find_peaks(
            voltage_filtered,
            height=MIN_PEAK_HEIGHT,
            distance=MIN_DISTANCE_SAMPLES
        )

        if len(peaks) > 0:
            last_peak_index = peaks[-1]
            
            # t[index] gives the time relative to the window (e.g., -1.2s)
            peak_time_in_window = t[last_peak_index] 
            time_of_latest_peak = current_time_s + peak_time_in_window
            
            # Check if a new, unique peak was found (time must be greater than last recorded time)
            if time_of_latest_peak > LAST_PEAK_TIME:
                
                rr_interval_s = time_of_latest_peak - LAST_PEAK_TIME
                
                # Validation: Check for a realistic RR interval (30 BPM to 240 BPM)
                if rr_interval_s > 0.25 and rr_interval_s < 2.0:
                    
                    CURRENT_BPM = 60.0 / rr_interval_s
                    LAST_PEAK_TIME = time_of_latest_peak
                    
                    # Update the BPM display
                    bpm_label.setText(
                        f"<span style='color: #FF69B4; font-size: 30pt; font-weight: bold;'>BPM: {CURRENT_BPM:.1f}</span>"
                    )

        # 4. Update the PyQTGraph plot
        curve.setData(t,voltage_filtered)


# ============================================================
# 5. TIMER SETUP and EXECUTION
# ============================================================
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10) # Update every 10 milliseconds

print(f"Listening for ECG data on UDP port {UDP_PORT}. Plotting {WINDOW_SEC} seconds at {FS} Hz.")
app.exec_()