import socket
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

# ============================================================
#               USER PARAMETERS
# ============================================================
UDP_PORT = 12345
FS = 500               # sampling rate from Beagle
WINDOW_SEC = 5         # how many seconds to display
N = FS * WINDOW_SEC    # total buffer size

BEAGLE_IP = "192.168.7.2" 


# ============================================================
#                  1. SET UP UDP SOCKET
# ============================================================

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(("0.0.0.0", UDP_PORT))

# TODO: Make socket non-blocking
sock.setblocking(False)

sock.sendto(b"send\n", (BEAGLE_IP, UDP_PORT))

# ============================================================
#                  2. SET UP DATA BUFFER
# ============================================================


buffer = np.zeros(N, dtype=float)

print(">>> Running ECG plot script version X")


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
    # Try reading all available UDP packets (non-blocking mode)
    # --------------------------------------------------------
    try:
        while True:
            data, addr = sock.recvfrom(1024)
            text = data.decode().strip()

            # convert text to float sample
            try:
                v = float(text)
                print("got:", v)
            except ValueError:
                # ignore malformed packets
                continue

            # shift buffer left by 1 and insert new value at end
            buffer = np.roll(buffer, -1)
            buffer[-1] = v

            

    except BlockingIOError:
        # No more packets this frame → continue to drawing
        pass

    # --------------------------------------------------------
    # Update plot line
    # --------------------------------------------------------
    curve.setData(t, buffer)


# ============================================================
#           5. SET UP QT TIMER TO CALL update()
# ============================================================

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(10)          # update every 10 ms


# ============================================================
#           6. START APPLICATION EVENT LOOP
# ============================================================

print("Ready to receive ECG data over UDP port", UDP_PORT)
app.exec_()
