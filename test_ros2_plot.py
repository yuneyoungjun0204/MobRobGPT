#!/usr/bin/env python3
"""ROS2 GPS Data Receiver + Real-time Plot Test.

Plots latitude/longitude from GPS topics (NavSatFix) in meters.
Uses absolute coordinate system identical to usv-simulator battlefield view.

Usage:
    python test_ros2_plot.py
    python test_ros2_plot.py --world 33    # 33m world
    python test_ros2_plot.py --world 12600 # 12.6km world
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import threading
import math


def _arg(flag, default=None):
    if flag in sys.argv:
        i = sys.argv.index(flag)
        return sys.argv[i + 1] if i + 1 < len(sys.argv) else default
    return default


# World size (configurable via --world flag)
WORLD_SIZE = float(_arg("--world", "33"))  # default 33m


# ============ UTM Conversion Functions ============
def latlon_to_utm(lat, lon):
    """WGS84 lat/lon to UTM (easting, northing) in meters.

    Returns: (easting, northing, zone_number, zone_letter)
    """
    if not (-80.0 <= lat <= 84.0):
        return None  # Outside UTM range
    if not (-180.0 <= lon <= 180.0):
        return None

    # UTM zone 계산
    zone_number = int((lon + 180) / 6) + 1

    # Norway/Svalbard special cases
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        zone_number = 32
    if 72.0 <= lat < 84.0:
        if 0.0 <= lon < 9.0:
            zone_number = 31
        elif 9.0 <= lon < 21.0:
            zone_number = 33
        elif 21.0 <= lon < 33.0:
            zone_number = 35
        elif 33.0 <= lon < 42.0:
            zone_number = 37

    # Zone letter
    if 84 >= lat >= 72: zone_letter = 'X'
    elif 72 > lat >= 64: zone_letter = 'W'
    elif 64 > lat >= 56: zone_letter = 'V'
    elif 56 > lat >= 48: zone_letter = 'U'
    elif 48 > lat >= 40: zone_letter = 'T'
    elif 40 > lat >= 32: zone_letter = 'S'
    elif 32 > lat >= 24: zone_letter = 'R'
    elif 24 > lat >= 16: zone_letter = 'Q'
    elif 16 > lat >= 8: zone_letter = 'P'
    elif 8 > lat >= 0: zone_letter = 'N'
    elif 0 > lat >= -8: zone_letter = 'M'
    elif -8 > lat >= -16: zone_letter = 'L'
    elif -16 > lat >= -24: zone_letter = 'K'
    elif -24 > lat >= -32: zone_letter = 'J'
    elif -32 > lat >= -40: zone_letter = 'H'
    elif -40 > lat >= -48: zone_letter = 'G'
    elif -48 > lat >= -56: zone_letter = 'F'
    elif -56 > lat >= -64: zone_letter = 'E'
    elif -64 > lat >= -72: zone_letter = 'D'
    elif -72 > lat >= -80: zone_letter = 'C'
    else: zone_letter = 'Z'  # Out of range

    # WGS84 파라미터
    a = 6378137.0  # semi-major axis
    f = 1 / 298.257223563  # flattening
    k0 = 0.9996  # scale factor
    e = math.sqrt(2 * f - f * f)  # eccentricity
    e2 = e * e / (1 - e * e)  # second eccentricity squared

    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    lon0 = math.radians((zone_number - 1) * 6 - 180 + 3)  # central meridian

    N = a / math.sqrt(1 - e * e * math.sin(lat_rad) ** 2)
    T = math.tan(lat_rad) ** 2
    C = e2 * math.cos(lat_rad) ** 2
    A = math.cos(lat_rad) * (lon_rad - lon0)

    M = a * ((1 - e * e / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256) * lat_rad
             - (3 * e * e / 8 + 3 * e ** 4 / 32 + 45 * e ** 6 / 1024) * math.sin(2 * lat_rad)
             + (15 * e ** 4 / 256 + 45 * e ** 6 / 1024) * math.sin(4 * lat_rad)
             - (35 * e ** 6 / 3072) * math.sin(6 * lat_rad))

    easting = k0 * N * (A + (1 - T + C) * A ** 3 / 6
                        + (5 - 18 * T + T * T + 72 * C - 58 * e2) * A ** 5 / 120) + 500000

    northing = k0 * (M + N * math.tan(lat_rad) * (
        A * A / 2 + (5 - T + 9 * C + 4 * C * C) * A ** 4 / 24
        + (61 - 58 * T + T * T + 600 * C - 330 * e2) * A ** 6 / 720))

    if lat < 0:
        northing += 10000000  # Southern hemisphere offset

    return (easting, northing, zone_number, zone_letter)


def latlon_to_local_meters(lat, lon, origin_lat, origin_lon):
    """Simple local coordinate conversion (meters from origin).

    Simpler than UTM for small areas.
    """
    # 1 degree latitude ≈ 111,320m
    # 1 degree longitude ≈ 111,320m * cos(lat)
    d_lat = lat - origin_lat
    d_lon = lon - origin_lon

    y = d_lat * 111320.0  # north (meters)
    x = d_lon * 111320.0 * math.cos(math.radians(origin_lat))  # east (meters)

    return (x, y)

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    from sensor_msgs.msg import NavSatFix
    ROS2_OK = True
except ImportError:
    ROS2_OK = False
    print("rclpy not found - ROS2 test unavailable")
    exit(1)


class GPSPlotter(Node):
    def __init__(self):
        super().__init__('gps_plotter')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Data storage
        self.mothership = None
        self.allies = [None, None, None]
        self.enemies = [None] * 10
        self.ally_history = [deque(maxlen=50) for _ in range(3)]
        self.enemy_history = [deque(maxlen=50) for _ in range(10)]
        self.lock = threading.Lock()

        # GPS subscriptions
        self.create_subscription(NavSatFix, '/mothership/fix', self._on_mother, qos)
        for i in range(3):
            self.create_subscription(
                NavSatFix, f'/ally_{i}/fix',
                lambda msg, idx=i: self._on_ally(idx, msg), qos)
        for i in range(10):
            self.create_subscription(
                NavSatFix, f'/enemy_{i}/fix',
                lambda msg, idx=i: self._on_enemy(idx, msg), qos)

        print("GPS Plotter started - subscribing to topics...")
        print("  /mothership/fix")
        print("  /ally_0/fix ~ /ally_2/fix")
        print("  /enemy_0/fix ~ /enemy_9/fix")

    def _on_mother(self, msg):
        with self.lock:
            # latitude/longitude 값을 그대로 x/y로 사용
            self.mothership = (msg.latitude, msg.longitude)

    def _on_ally(self, idx, msg):
        with self.lock:
            pos = (msg.latitude, msg.longitude)
            self.allies[idx] = pos
            self.ally_history[idx].append(pos)

    def _on_enemy(self, idx, msg):
        with self.lock:
            pos = (msg.latitude, msg.longitude)
            self.enemies[idx] = pos
            self.enemy_history[idx].append(pos)

    def get_data(self):
        with self.lock:
            return {
                'mothership': self.mothership,
                'allies': self.allies.copy(),
                'enemies': self.enemies.copy(),
                'ally_history': [list(h) for h in self.ally_history],
                'enemy_history': [list(h) for h in self.enemy_history],
            }


def main():
    rclpy.init()
    node = GPSPlotter()

    # ROS2 spin thread
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()

    # Coordinate system: 0 ~ world_size, center = (world_size/2, world_size/2)
    # X=North, Y=East
    W = WORLD_SIZE
    half = W / 2

    # 1.25x margin for view
    margin = W * 0.125
    view_min = -margin
    view_max = W + margin

    # Matplotlib plot
    fig, ax2 = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle(f'Battlefield Status (world={W}m)\n[Scroll: zoom] [Middle drag: pan] [r: reset view]')

    # Zoom/pan state storage with 1.25x margin
    view_state = {
        'xlim': (view_min, view_max), 'ylim': (view_min, view_max),
    }

    def on_scroll(event):
        """Mouse wheel zoom in/out."""
        if event.inaxes is None:
            return
        scale = 1.2 if event.button == 'down' else 1/1.2

        xlim = ax2.get_xlim()
        ylim = ax2.get_ylim()
        xdata, ydata = event.xdata, event.ydata

        new_xlim = [xdata - (xdata - xlim[0]) * scale,
                    xdata + (xlim[1] - xdata) * scale]
        new_ylim = [ydata - (ydata - ylim[0]) * scale,
                    ydata + (ylim[1] - ydata) * scale]

        ax2.set_xlim(new_xlim)
        ax2.set_ylim(new_ylim)
        view_state['xlim'] = new_xlim
        view_state['ylim'] = new_ylim

        fig.canvas.draw_idle()

    def on_key(event):
        """Reset view with 'r' key."""
        if event.key == 'r':
            view_state['xlim'] = (view_min, view_max)
            view_state['ylim'] = (view_min, view_max)
            print("[View Reset]")

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', on_key)

    # Pan (drag) state
    pan_state = {'pressed': False, 'x': 0, 'y': 0, 'ax': None}

    def on_press(event):
        if event.button == 2 and event.inaxes:  # Middle button
            pan_state['pressed'] = True
            pan_state['x'] = event.xdata
            pan_state['y'] = event.ydata
            pan_state['ax'] = event.inaxes

    def on_release(event):
        pan_state['pressed'] = False

    def on_motion(event):
        if not pan_state['pressed'] or event.inaxes != pan_state['ax']:
            return
        dx = pan_state['x'] - event.xdata
        dy = pan_state['y'] - event.ydata

        xlim = ax2.get_xlim()
        ylim = ax2.get_ylim()
        ax2.set_xlim(xlim[0] + dx, xlim[1] + dx)
        ax2.set_ylim(ylim[0] + dy, ylim[1] + dy)
        view_state['xlim'] = ax2.get_xlim()
        view_state['ylim'] = ax2.get_ylim()

        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)

    def update(_):
        data = node.get_data()

        # Coordinate system: X=North, Y=East
        ax2.clear()
        ax2.set_title(f'Battlefield Status (world={W}m)')
        ax2.set_xlabel('North (m)')
        ax2.set_ylabel('East (m)')
        ax2.set_xlim(view_state['xlim'])
        ax2.set_ylim(view_state['ylim'])
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.3)

        # World boundary
        ax2.plot([0, W, W, 0, 0], [0, 0, W, W, 0], 'k--', lw=1, alpha=0.5)

        if data['mothership']:
            # GPS coordinates (publisher sends in correct order)
            origin_lat, origin_lon = data['mothership']  # Normal order

            # Mothership position = center (world_size/2, world_size/2)
            mx, my = half, half
            ax2.scatter(mx, my, s=200, c='gold', marker='*', label='Mothership', zorder=10)

            # Center lines
            ax2.axhline(half, color='gray', lw=0.5, alpha=0.3)
            ax2.axvline(half, color='gray', lw=0.5, alpha=0.3)

            # Allies - GPS to absolute coordinate conversion (180 degree rotation)
            for i, pos in enumerate(data['allies']):
                if pos:
                    lat, lon = pos  # Normal order
                    east, north = latlon_to_local_meters(lat, lon, origin_lat, origin_lon)
                    # X=North, Y=East (180 degree rotation: sign inversion)
                    x, y = half - north, half - east
                    ax2.scatter(x, y, s=100, c='#42A5F5', marker='o', edgecolors='white', linewidths=1, zorder=5)
                    ax2.annotate(f'A{i}', (x, y + 0.8), fontsize=8, color='#42A5F5', ha='center')

            # Enemies - GPS to absolute coordinate conversion (180 degree rotation)
            for i, pos in enumerate(data['enemies']):
                if pos:
                    lat, lon = pos  # Normal order
                    east, north = latlon_to_local_meters(lat, lon, origin_lat, origin_lon)
                    # X=North, Y=East (180 degree rotation: sign inversion)
                    x, y = half - north, half - east
                    ax2.scatter(x, y, s=80, c='#EF5350', marker='^', edgecolors='white', linewidths=0.5, zorder=5)
                    ax2.annotate(f'E{i}', (x, y + 0.8), fontsize=7, color='#EF5350', ha='center')

        ax2.legend(loc='upper right', fontsize=8)

        return []

    anim = FuncAnimation(fig, update, interval=100, blit=False)

    print(f"\nPlot window opened (world={W}m, center=({half}, {half}))")
    print("Same coordinate system as usv-simulator battlefield view. Close window to exit.")
    try:
        plt.show()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
