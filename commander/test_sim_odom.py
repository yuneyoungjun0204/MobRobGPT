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


@unittest.skipUnless(CKPT.exists(), "u-net_map.pt 없음")
class TestMinNetLen(unittest.TestCase):
    """미니 수조 규모(span 43 m)에서 최소 그물 길이 강제 — 중점 고정·양쪽 균등 확장."""

    SPAN = 43.0
    K = 43.0 / 4400.0          # 녹화본 배치(≈±2 km)를 수조 규모로 축소

    def _env(self):
        from commander.sim_odom_env import SimOdomCnnEnv
        st = SimOdomState(stale_timeout=1.0, clock=lambda: 0.0)
        for sid, (x, y, yaw) in SCENE.items():
            st.on_odom(sid, x * self.K, y * self.K, yaw)
        av = SimShipView(st, ALLIES, drop_captured=False)
        ev = SimShipView(st, ENEMIES, drop_captured=True)
        return SimOdomCnnEnv(str(CKPT), self.SPAN, av, ev,
                             waypoint_sink=SimWaypointSink(verbose=False),
                             net_sink=SimNetTriggerSink(verbose=False),
                             publish_hz=1e6, enu_origin=(0.0, 0.0))

    def _decide(self, env, min_len_m):
        from commander.fallback import heuristic_plan
        from commander.rl_bridge import build_battlefield_defense
        env.step()                                   # 첫 수신 → ready
        env.set_plan(heuristic_plan(build_battlefield_defense(env)))
        env.set_min_net_len_real(min_len_m)
        env._rl_decide()

    def _legs(self, env):
        out = {}
        for p in range(env.P):
            if env._assign[0, p] >= 0 and env.net_mask[0, p, 1]:
                a, b = env.route[0, p, 0].copy(), env.route[0, p, 1].copy()
                out[p] = (a, b)
        return out

    def test_extends_symmetrically(self):
        base = self._env()
        self._decide(base, 0.0)
        before = self._legs(base)
        self.assertTrue(before)
        L_m = 3.0
        env = self._env()
        self._decide(env, L_m)
        after = self._legs(env)
        L = env.min_net_len_sim
        for p, (a0, b0) in before.items():
            a1, b1 = after[p]
            d0, d1 = np.hypot(*(b0 - a0)), np.hypot(*(b1 - a1))
            if d0 >= L:
                np.testing.assert_allclose([a1, b1], [a0, b0])          # 이미 길면 그대로
                continue
            self.assertAlmostEqual(d1, L, delta=1e-6)                   # 정확히 하한 길이
            np.testing.assert_allclose((a1 + b1) / 2, (a0 + b0) / 2, atol=1e-6)   # 중점 고정
            if d0 > 1e-6:                                               # 방향 유지
                np.testing.assert_allclose((b1 - a1) / d1, (b0 - a0) / d0, atol=1e-6)
            np.testing.assert_allclose(env.route[0, p, 2:], np.broadcast_to(b1, env.route[0, p, 2:].shape))
        self.assertAlmostEqual(env.net_len_cap(), max(env.cfg.net_max_len, L))
        # 실미터 기준으로도 3 m
        p = next(iter(after))
        a1, b1 = (env.scale.sim_to_enu(x) for x in after[p])
        self.assertAlmostEqual(float(np.hypot(*(b1 - a1))), L_m, delta=1e-6)

    def test_degenerate_points_use_tangent(self):
        env = self._env()
        self._decide(env, 0.0)
        p = next(iter(self._legs(env)))
        mid = env.route[0, p, 0].copy()
        env.route[0, p, 1] = mid                                        # 두 점이 겹친 경우
        env.set_min_net_len_real(2.0)
        env._enforce_min_net_len()
        a, b = env.route[0, p, 0], env.route[0, p, 1]
        np.testing.assert_allclose((a + b) / 2, mid, atol=1e-6)
        radial = mid - np.asarray(env.center)
        self.assertAlmostEqual(float(np.dot(b - a, radial)), 0.0, delta=1e-3)   # 접선 방향

    def test_skips_ship_already_netting(self):
        env = self._env()
        self._decide(env, 0.0)
        p = next(iter(self._legs(env)))
        env.doing_net[0, p] = True
        before = env.route[0, p].copy()
        env.set_min_net_len_real(5.0)
        env._enforce_min_net_len()
        np.testing.assert_allclose(env.route[0, p], before)

    def test_off_by_default(self):
        env = self._env()
        self.assertEqual(env.min_net_len_sim, 0.0)
        self.assertEqual(env.net_len_cap(), env.cfg.net_max_len)


@unittest.skipUnless(CKPT.exists(), "u-net_map.pt 없음")
class TestArriveRadiusMult(unittest.TestCase):
    def test_mult_widens_arrival_only(self):
        env = TestMinNetLen()._env()
        base = env.arrive_r()
        self.assertEqual(base, env.cfg.arrive_radius)            # 기본 1배 = 학습값
        env.arrive_radius_mult = 3.0
        self.assertAlmostEqual(env.arrive_r(), 3.0 * env.cfg.arrive_radius)
        self.assertEqual(env.cfg.arrive_radius, base)             # cfg(배정 링 하한 등)는 그대로
        # 3배 반경 안·1배 반경 밖에 있는 아군 → 도착으로 판정돼 WP 가 전진해야 한다
        TestMinNetLen()._decide(env, 0.0)
        p = next(p for p in range(env.P) if env._assign[0, p] >= 0)
        wp = env.route[0, p, 0].copy()
        env.ptr[0, p] = 0
        env.a_pos[0, p] = wp + np.array([2.0 * env.cfg.arrive_radius, 0.0])
        env._prev_pos = env.a_pos[0].copy()
        env._advance_and_paint()
        self.assertGreaterEqual(int(env.ptr[0, p]) + int(env.doing_net[0, p]), 1)

    def test_set_arrive_radius_real(self):
        env = TestMinNetLen()._env()
        env.set_arrive_radius_real(0.5)
        self.assertAlmostEqual(float(env.scale.len_to_real(env.arrive_r())), 0.5, places=9)
        with self.assertRaises(ValueError):
            env.set_arrive_radius_real(0.0)


@unittest.skipUnless(CKPT.exists(), "u-net_map.pt 없음")
class TestRestartAndRescale(unittest.TestCase):
    """space 초기화(restart_episode)·span 슬라이더(rescale) — 프로세스 재시작 없이 새 에피소드."""

    def _run_some(self, env, steps=60):
        from commander.fallback import heuristic_plan
        from commander.rl_bridge import build_battlefield_defense
        for i in range(steps):
            env.step()
            if i == 1:
                env.set_plan(heuristic_plan(build_battlefield_defense(env)))

    def test_restart_clears_episode_state(self):
        env = TestMinNetLen()._env()
        self._run_some(env)
        # 그물·격침·소진 상태를 인위로 만들어 둔다
        env.painted[0][:5, :5] = 1
        env.net_installed[0][:5, :5] = True
        env.doing_net[0, 0] = True
        env.a_nets[0, 1] = 0
        env.stats["captures"] = 4
        self.assertTrue(env.ready)
        env.restart_episode()
        self.assertFalse(env.painted[0].any())
        self.assertFalse(env.net_installed[0].any())
        self.assertFalse(env.doing_net[0].any())
        self.assertTrue((env.a_nets[0] == env.cfg.nets_per_ship).all())
        self.assertEqual(env.stats["captures"], 0)
        self.assertIsNone(env._plan)
        self.assertFalse(env.ready)                 # 텔레메트리 재수신(쿼럼)부터 다시
        self.assertEqual(env._micro_ct, 0)
        self.assertEqual(env.bag_time_real, 0.0)
        env.step()                                  # 다음 tick 에 다시 붙는다
        self.assertTrue(env.ready)

    def test_rescale_updates_scale_and_min_net(self):
        env = TestMinNetLen()._env()
        env.set_min_net_len_real(9.5)
        s0, l0 = env.scale.S, env.min_net_len_sim
        env.arrive_radius_mult = 4.2
        env.rescale(70.0)
        self.assertAlmostEqual(env.scale.S, env.cfg.world_size / 70.0)
        self.assertAlmostEqual(float(env.scale.len_to_real(env.min_net_len_sim)), 9.5)   # 실미터 유지
        self.assertNotAlmostEqual(env.min_net_len_sim, l0)
        self.assertEqual(env.arrive_radius_mult, 4.2)       # 배수 의미는 유지
        self.assertNotAlmostEqual(env.scale.S, s0)
