"""Tests for the simulator (ROS2 Odometry, local metre frame) adapter + env.

Run: python3.10 -m pytest -p no:anyio -q commander/test_sim_odom.py
(no rclpy needed — SimOdomState/sinks are ROS-free, the env is fed through them directly)
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from commander.sim_odom_ros2 import (SimNetTriggerSink, SimOdomState, SimShipView,
                                     SimWaypointSink, yaw_enu_from_quat, yaw_enu_to_nav)

CKPT = Path(__file__).resolve().parent.parent / "boatattack_sim/models/u-net_map.pt"
ALLIES = ["friendly_01", "friendly_02", "friendly_03"]
ENEMIES = [f"enemy_{i:02d}" for i in range(1, 8)]
# 2026-10-04 로스백(usv_20261004_194054) 의 running 직후 배치(x, y, ENU yaw)
SCENE = {
    "friendly_01": (48.0, 263.0, 90.0), "friendly_02": (260.0, -103.0, 0.0),
    "friendly_03": (-165.0, -109.0, 180.0),
    "enemy_01": (-42.8, 1870.0, -90.0), "enemy_02": (-538.0, -985.0, 60.0),
    "enemy_03": (-289.0, -1077.0, 75.0), "enemy_04": (130.0, -1068.0, 95.0),
    "enemy_05": (1846.0, -794.0, 157.0), "enemy_06": (1782.0, -932.0, 152.0),
    "enemy_07": (1758.0, -1140.0, 147.0),
}


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _state(clock=None) -> SimOdomState:
    st = SimOdomState(stale_timeout=1.0, clock=clock or _Clock())
    for sid, (x, y, yaw) in SCENE.items():
        st.on_odom(sid, x, y, yaw)
    return st


class TestConversions(unittest.TestCase):
    def test_yaw_enu_to_nav(self):
        self.assertAlmostEqual(yaw_enu_to_nav(90.0), 0.0)     # 북
        self.assertAlmostEqual(yaw_enu_to_nav(0.0), 90.0)     # 동
        self.assertAlmostEqual(yaw_enu_to_nav(-90.0), 180.0)  # 남
        self.assertAlmostEqual(yaw_enu_to_nav(180.0), 270.0)  # 서

    def test_quat_yaw(self):
        import math
        h = math.radians(157.0) / 2
        self.assertAlmostEqual(yaw_enu_from_quat(0, 0, math.sin(h), math.cos(h)), 157.0, places=6)


class TestState(unittest.TestCase):
    def test_stale_and_captured(self):
        clk = _Clock()
        st = _state(clk)
        st.on_sim_state({"sim_id": "run-1", "phase": "running",
                         "ships": [{"ship": "friendly_01", "nets_left": 2},
                                   {"ship": "enemy_04", "captured": True}]})
        ev = SimShipView(st, ENEMIES, drop_captured=True)
        self.assertEqual(ev.ally_snapshot().alive.tolist(),
                         [True, True, True, False, True, True, True])
        self.assertEqual(st.nets_left, {"friendly_01": 2})
        clk.t = 0.5
        st.on_odom("enemy_01", 0.0, 1800.0, -90.0)
        clk.t = 1.2                       # enemy_01 만 신선
        self.assertEqual(ev.ally_snapshot().alive.tolist(),
                         [True, False, False, False, False, False, False])
        av = SimShipView(st, ALLIES, drop_captured=False)
        self.assertFalse(av.ally_snapshot().alive.any())

    def test_heading_is_nav(self):
        st = _state()
        snap = SimShipView(st, ["enemy_01"], drop_captured=True).ally_snapshot()
        self.assertAlmostEqual(float(snap.hdg[0]), 180.0)     # ENU -90 = 남진


class TestSinks(unittest.TestCase):
    def test_waypoint_json_and_dedupe(self):
        out = []
        sink = SimWaypointSink(out.append, verbose=False)
        self.assertTrue(sink.publish("friendly_01", [(1.0, 2.0), (3.0, 4.0)], "run-1"))
        self.assertFalse(sink.publish("friendly_01", [(1.02, 2.0), (3.0, 4.0)], "run-1"))
        self.assertTrue(sink.publish("friendly_01", [], "run-1"))
        msg = json.loads(out[0])
        self.assertEqual(set(msg), {"sim_id", "ship", "points", "cruise_kn",
                                    "stop_at_final", "accept_radius_m"})
        self.assertEqual(msg["points"][1], {"x_m": 3.0, "y_m": 4.0})
        self.assertEqual(json.loads(out[1])["points"], [])

    def test_dry_run_records_without_publishing(self):
        sink = SimWaypointSink(None, verbose=False)
        sink.publish("friendly_01", [(1.0, 2.0)], None)
        self.assertEqual(len(sink.sent), 1)

    def test_net_trigger_only_on_start(self):
        out = []
        sink = SimNetTriggerSink(out.append, sim_id_fn=lambda: "run-1", verbose=False)
        sink.set("friendly_02", True)
        sink.set("friendly_02", False)
        self.assertEqual([json.loads(d) for d in out],
                         [{"sim_id": "run-1", "ship": "friendly_02", "lay": True}])


@unittest.skipUnless(CKPT.exists(), "u-net_map.pt 없음")
class TestSimOdomEnv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from commander.fallback import heuristic_plan
        from commander.rl_bridge import build_battlefield_defense
        from commander.sim_odom_env import SimOdomCnnEnv, auto_span
        from boatattack_sim.model.cnn_actor import load_cnn_actor

        cls.st = _state()
        cls.st.on_sim_state({"sim_id": "run-1", "ships": [{"ship": "enemy_04", "captured": True}]})
        av = SimShipView(cls.st, ALLIES, drop_captured=False)
        ev = SimShipView(cls.st, ENEMIES, drop_captured=True)
        _, cfg = load_cnn_actor(str(CKPT))
        cls.span = auto_span(cfg, 2000.0)
        cls.wp = SimWaypointSink(verbose=False)
        cls.net = SimNetTriggerSink(verbose=False)
        cls.env = SimOdomCnnEnv(str(CKPT), cls.span, av, ev, waypoint_sink=cls.wp,
                                net_sink=cls.net, sim_id_fn=lambda: "run-1",
                                publish_hz=1e6, enu_origin=(0.0, 0.0))
        env = cls.env
        for i in range(30):
            if i == 1:
                env.set_plan(heuristic_plan(build_battlefield_defense(env)))
            env.step()

    def test_auto_span(self):
        from commander.sim_odom_env import auto_span
        cfg = self.env.cfg
        self.assertAlmostEqual(auto_span(cfg, cfg.enemy_spawn_radius), cfg.world_size)

    def test_no_land(self):
        self.assertEqual(int(self.env.land_map.sum()), 0)

    def test_origin_maps_to_centre(self):
        c = self.env.scale.enu_to_sim(np.array([[0.0, 0.0]]))[0]
        np.testing.assert_allclose(c, [self.env.cfg.world_size / 2] * 2)
        np.testing.assert_allclose(self.env.scale.sim_to_enu(c), [0.0, 0.0], atol=1e-9)

    def test_ingest(self):
        env = self.env
        self.assertTrue(env.ready)
        np.testing.assert_allclose(env.scale.sim_to_enu(env.a_pos[0, 0]), SCENE["friendly_01"][:2])
        self.assertEqual(env.e_alive[0].tolist(),
                         [True, True, True, False, True, True, True, False, False, False])

    def test_waypoints_published_in_metres(self):
        env = self.env
        self.assertTrue(self.wp.sent)
        half = self.span / 2
        for msg in self.wp.sent:
            self.assertIn(msg["ship"], ALLIES)
            self.assertEqual(msg["sim_id"], "run-1")
            for pt in msg["points"]:
                self.assertLessEqual(abs(pt["x_m"]), half + 1e-6)
                self.assertLessEqual(abs(pt["y_m"]), half + 1e-6)
        for p in range(env.P):
            pts = env.waypoints_xy(p)
            if env._assign[0, p] >= 0:
                self.assertGreaterEqual(len(pts), 1)
                self.assertTrue(all(np.hypot(*np.subtract(pts[i], pts[i - 1])) > 1e-6
                                    for i in range(1, len(pts))))


if __name__ == "__main__":
    unittest.main()
