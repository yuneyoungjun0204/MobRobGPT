"""Tests for commander/gcs_bridge.py and commander/gcs_cnn_env.py.

Run: python3.10 -m unittest commander.test_gcs_bridge -v
(needs the same env as run_replay_infer.py: torch + numpy, no rclpy/ROS2 required --
these tests never touch bag_replay.py's rosbag2 loader, and never talk to a real GCS.)
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from commander.gcs_bridge import AllySnapshot, GcsAllyLink, GcsClient, NetDeploySink

CKPT = os.path.join(os.path.dirname(__file__), "..",
                     "boatattack_sim", "models", "u-net_map.pt")


class _StubGcsHandler(BaseHTTPRequestHandler):
    """Records every request; answers /api/state with server.state and every
    goto POST with a fixed accepted verdict."""

    def log_message(self, *a):   # silence -- keep test output readable
        pass

    def do_GET(self):
        if self.path == "/api/state":
            body = json.dumps(self.server.state_json).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        self.server.posts.append((self.path, payload))
        verdict = {"accepted": True, "reason": "ok", "detail": ""}
        body = json.dumps(verdict).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


class _StubGcsServer:
    """A real (loopback) HTTP server so GcsClient's urllib path is exercised end-to-end."""

    def __init__(self):
        self.httpd = HTTPServer(("127.0.0.1", 0), _StubGcsHandler)
        self.httpd.state_json = {"t": 0.0, "vehicles": {}}
        self.httpd.posts = []
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base_url(self) -> str:
        host, port = self.httpd.server_address
        return f"http://{host}:{port}"

    @property
    def posts(self):
        return self.httpd.posts

    def set_state(self, vehicles: dict) -> None:
        self.httpd.state_json = {"t": 0.0, "vehicles": vehicles}

    def close(self):
        self.httpd.shutdown()
        self.thread.join(timeout=2.0)
        self.httpd.server_close()


class TestGcsClient(unittest.TestCase):
    def setUp(self):
        self.server = _StubGcsServer()
        self.addCleanup(self.server.close)
        self.client = GcsClient(self.server.base_url, timeout=2.0)

    def test_get_json_roundtrip(self):
        self.server.set_state({"usv1": {"ned": {"x": 1.0, "y": 2.0}, "heading": 90.0,
                                        "connected": True, "stale": {"position": False}}})
        got = self.client.get_json("/api/state")
        self.assertEqual(got["vehicles"]["usv1"]["ned"], {"x": 1.0, "y": 2.0})

    def test_post_json_reaches_server_and_returns_verdict(self):
        verdict = self.client.post_json(
            "/api/command/usv1/goto",
            {"east": 5.0, "north": -1.0, "source": "rl", "stamp": 123.0})
        self.assertTrue(verdict["accepted"])
        self.assertEqual(self.server.posts[-1][0], "/api/command/usv1/goto")
        self.assertEqual(self.server.posts[-1][1]["east"], 5.0)
        self.assertEqual(self.server.posts[-1][1]["source"], "rl")


class TestGcsAllyLink(unittest.TestCase):
    def test_ally_snapshot_maps_ned_to_east_north(self):
        client = GcsClient("http://unused.invalid")   # not called: state is passed in
        link = GcsAllyLink(client, ["usv1", "usv2"])
        state = {
            "vehicles": {
                # ned = {x: north, y: east} -- gcs's own convention (common/state.py)
                "usv1": {"ned": {"x": 10.0, "y": 20.0}, "heading": 45.0,
                         "connected": True, "stale": {"position": False}},
                "usv2": None,
            }
        }
        snap = link.ally_snapshot(state)
        self.assertIsInstance(snap, AllySnapshot)
        np.testing.assert_allclose(snap.pos[0], [20.0, 10.0])   # [east, north]
        self.assertAlmostEqual(snap.hdg[0], 45.0)
        self.assertTrue(snap.alive[0])
        self.assertFalse(snap.alive[1])                          # unregistered vehicle

    def test_ally_snapshot_treats_stale_or_disconnected_as_not_alive(self):
        client = GcsClient("http://unused.invalid")
        link = GcsAllyLink(client, ["usv1"])
        state = {"vehicles": {"usv1": {"ned": {"x": 0.0, "y": 0.0}, "heading": 0.0,
                                       "connected": False, "stale": {"position": False}}}}
        self.assertFalse(link.ally_snapshot(state).alive[0])

    def test_submit_goto_posts_expected_body(self):
        server = _StubGcsServer()
        self.addCleanup(server.close)
        link = GcsAllyLink(GcsClient(server.base_url), ["usv1"], default_speed=1.5)
        verdict = link.submit_goto("usv1", east=3.0, north=4.0, stamp=1000.0)
        self.assertTrue(verdict["accepted"])
        path, body = server.posts[-1]
        self.assertEqual(path, "/api/command/usv1/goto")
        self.assertEqual(body["east"], 3.0)
        self.assertEqual(body["north"], 4.0)
        self.assertEqual(body["source"], "rl")
        self.assertEqual(body["speed"], 1.5)


class TestNetDeploySink(unittest.TestCase):
    def test_set_logs_and_appends_file(self):
        lines = []
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "net.jsonl")
            sink = NetDeploySink(log_fn=lines.append, log_path=path)
            sink.set("usv1", True, stamp=1.0)
            sink.set("usv1", False, stamp=2.0)
            with open(path, encoding="utf-8") as f:
                recs = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(recs, [
            {"vehicle_id": "usv1", "net_deploy": True, "stamp": 1.0},
            {"vehicle_id": "usv1", "net_deploy": False, "stamp": 2.0},
        ])

    def test_set_never_mentions_actuator_tokens(self):
        import inspect
        from commander import gcs_bridge
        src = inspect.getsource(gcs_bridge)
        forbidden = {"mav.", "_send(", "MAV_CMD", "set_actuator",
                     "servo", "relay", "pymavlink"}
        hit = [tok for tok in forbidden if tok.lower() in src.lower()]
        self.assertEqual(hit, [], f"gcs_bridge.py must never approach MAVLink/actuator APIs: {hit}")


@unittest.skipUnless(os.path.exists(CKPT), f"checkpoint not found: {CKPT}")
class TestGcsBagCnnEnvWaypointSequencing(unittest.TestCase):
    """Drives the new arrival/net-drop/publish state machine directly (bypassing the RL/LLM
    decision code, which is untouched, pre-existing behaviour) with a fully scripted ally
    trajectory, so the test is deterministic regardless of what the trained policy would
    have picked."""

    class _FakeBag:
        duration_sec = 1e9
        first_common_time_sec = 0.0

        def sample(self, t):
            return (np.zeros((0, 2)), np.zeros((0,)), np.zeros((0,), dtype=bool))

        def centroid(self):
            return (0.0, 0.0)

        def max_radius_from(self, origin):
            return 1.0

        def mean_speed_mps(self):
            return 1.0

    class _FakeAllyLink:
        def __init__(self, vehicle_ids):
            self.vehicle_ids = vehicle_ids
            self.calls = []
            self.pos = np.zeros((len(vehicle_ids), 2))
            self.alive = np.zeros(len(vehicle_ids), dtype=bool)
            self.raise_on_snapshot = False   # simulates an HTTP-level poll failure

        @property
        def n_allies(self):
            return len(self.vehicle_ids)

        def ally_snapshot(self, state=None):
            if self.raise_on_snapshot:
                raise RuntimeError("simulated GCS poll failure")
            # Match the real GcsAllyLink: a non-alive row is zeroed, never a stale
            # coordinate carried forward by the *link* itself (holding last-known is
            # GcsBagCnnEnv._ingest_gcs_allies's job, not the link's -- conflating the
            # two here is exactly what let bug B1 hide behind 9 green tests).
            pos = np.where(self.alive[:, None], self.pos, 0.0)
            return AllySnapshot(pos=pos, hdg=np.zeros(len(self.vehicle_ids)),
                                alive=self.alive.copy())

        def submit_goto(self, vehicle_id, east, north, speed=None, stamp=None):
            self.calls.append((vehicle_id, float(east), float(north)))
            return {"accepted": True, "reason": "ok", "detail": ""}

    class _RecordingSink:
        def __init__(self):
            self.events = []

        def set(self, vehicle_id, active, stamp=None):
            self.events.append((vehicle_id, bool(active)))

    def _build_env(self):
        from commander.gcs_cnn_env import GcsBagCnnEnv
        ally_link = self._FakeAllyLink(["usv1", "usv2", "usv3"])
        sink = self._RecordingSink()
        # span_real=63 with this checkpoint's world_size=12600, arrive_radius=200 sim-m
        # makes arrive_radius exactly 1.0 real metre, and net_max_len=450 sim-m -> 2.25 m.
        env = GcsBagCnnEnv(
            CKPT, self._FakeBag(), span_real=63.0, ally_link=ally_link,
            net_sink=sink, publish_hz=1e9, nets_per_ship=1)
        self.assertEqual(env.P, 3, "test assumes the shipped u-net_map.pt has P=3 allies")
        return env, ally_link, sink

    def _prime_ship0(self, env, wp1_enu, wp2_enu):
        wp1_sim = env.scale.enu_to_sim(np.asarray(wp1_enu, dtype=np.float64))
        wp2_sim = env.scale.enu_to_sim(np.asarray(wp2_enu, dtype=np.float64))
        env.a_nets[0, 0] = 1
        env._assign[0, 0] = 0
        env.a_alive[0, 0] = True
        env.route[0, 0, 0] = wp1_sim
        env.route[0, 0, 1] = wp2_sim
        for k in range(2, env.Kw):
            env.route[0, 0, k] = wp2_sim
        env.net_mask[0, 0, :] = False
        env.net_mask[0, 0, 1] = True
        env.ptr[0, 0] = 0
        env.doing_net[0, 0] = False
        env.leg_netted[0, 0] = False
        env.paint_dist[0, 0] = 0.0

    @staticmethod
    def _tick(env, ally_link, east):
        """Drives the sub-methods `step()` normally sequences, for ship0 only -- so a
        test can script its trajectory tick-by-tick without depending on the trained
        policy's own pixel choice. `_t_real` is advanced manually because `step()`
        itself is bypassed here."""
        ally_link.pos[0] = (east, 0.0)
        env._ingest_gcs_allies()
        env._advance_and_paint()
        env._t_real += env.scale.dt_real
        env._publish_to_gcs()

    def test_wp1_then_net_start_then_wp2_then_net_finish(self):
        env, ally_link, sink = self._build_env()
        wp1 = (10.0, 0.0)
        wp2 = (30.0, 0.0)
        self._prime_ship0(env, wp1, wp2)
        ally_link.alive[:] = [True, False, False]   # only usv1 is under test

        # Far from wp1: no net event yet, and the published target must be wp1.
        self._tick(env, ally_link, 0.0)
        self.assertEqual(sink.events, [])
        self.assertAlmostEqual(ally_link.calls[-1][1], 10.0, places=3)   # east
        self.assertAlmostEqual(ally_link.calls[-1][2], 0.0, places=3)    # north

        # Within arrive_radius (1.0 m) of wp1: ptr advances, but net does not start
        # on the SAME tick as arrival (matches defense_env.py's _micro ordering).
        self._tick(env, ally_link, 9.5)
        self.assertEqual(sink.events, [], "net must not start before arrival is registered")

        # One more tick at the same spot: now on the net leg -> net starts.
        self._tick(env, ally_link, 9.5)
        self.assertEqual(sink.events, [("usv1", True)])
        # The bridge must now be commanding wp2, not wp1.
        self.assertAlmostEqual(ally_link.calls[-1][1], 30.0, places=3)

        # Walk toward wp2 in small steps; net_max_len (2.25 m here) must end the leg
        # long before the ship geometrically reaches wp2 (20 m away).
        east = 9.5
        finished = False
        for _ in range(20):
            east += 0.3
            self._tick(env, ally_link, east)
            if ("usv1", False) in sink.events:
                finished = True
                break
        self.assertTrue(finished, "net leg never finished within net_max_len budget")
        self.assertEqual(sink.events, [("usv1", True), ("usv1", False)])

    def test_stale_ally_holds_last_known_position_not_phantom_origin(self):
        """Regression for architect-review finding B1: a link drop must hold the ship's
        last known position, never fall back to enu_to_sim((0,0)) -- an arbitrary point
        inside the map that would corrupt this ship's own arrival check and feed a
        phantom position into the fleet's shared clustering/assignment observation."""
        env, ally_link, sink = self._build_env()
        self._prime_ship0(env, (10.0, 0.0), (30.0, 0.0))
        ally_link.alive[:] = [True, False, False]

        self._tick(env, ally_link, 3.0)
        held_pos = env.a_pos[0, 0].copy()
        np.testing.assert_allclose(held_pos, env.scale.enu_to_sim(np.array([3.0, 0.0])))
        calls_before = len(ally_link.calls)

        # Link drops: gcs no longer reports this vehicle as connected/fresh.
        ally_link.alive[0] = False
        self._tick(env, ally_link, 3.0)   # ally_link.pos[0] is irrelevant now -- alive=False
                                          # must mask it out

        np.testing.assert_allclose(env.a_pos[0, 0], held_pos,
                                   err_msg="a_pos must hold the last known value")
        phantom = env.scale.enu_to_sim(np.array([0.0, 0.0]))
        self.assertFalse(np.allclose(env.a_pos[0, 0], phantom),
                         "a_pos must never fall back to the ENU-origin phantom point")
        self.assertFalse(bool(env.a_alive[0, 0]))
        self.assertEqual(len(ally_link.calls), calls_before,
                         "a ship gcs cannot currently locate must not be commanded")
        self.assertEqual(sink.events, [], "a dead ship must not register a spurious arrival")

    def test_dead_or_unregistered_ally_is_never_commanded(self):
        env, ally_link, sink = self._build_env()
        self._prime_ship0(env, (10.0, 0.0), (30.0, 0.0))
        ally_link.alive[:] = [False, False, False]
        ally_link.pos[0] = (0.0, 0.0)
        env._ingest_gcs_allies()
        env._advance_and_paint()
        env._publish_to_gcs()
        self.assertEqual(ally_link.calls, [])
        self.assertEqual(sink.events, [])

    def test_step_ticks_correctly_across_quorum_wait_and_a_later_outage(self):
        """Regression for the architect-review finding that three consecutive fix
        rounds each broke or nearly broke run_gcs_bridge.py's decision-counting contract
        with GcsBagCnnEnv.step() -- every other test in this file drives the private
        sub-methods directly and so never exercised step() itself. The contract:
        `_micro_ct` must not advance on ANY tick that does not run a real decision,
        whether that's the initial quorum wait or a later mid-run telemetry outage, and
        must advance by exactly one on the tick quorum is first reached.
        """
        env, ally_link, sink = self._build_env()
        ally_link.alive[:] = [True, True, False]   # usv3 hasn't joined yet
        ally_link.pos[:] = 0.0

        for _ in range(5):
            env.step()
            self.assertFalse(env.ready)
            self.assertEqual(env._micro_ct, 0,
                             "no decision may run before every hull has been seen once")
            self.assertEqual(env.missing_ally_ids(), ["usv3"])

        ally_link.alive[2] = True   # usv3 joins
        env.step()
        self.assertTrue(env.ready)
        self.assertEqual(env._micro_ct, 1,
                         "the transition tick itself must still count as one real tick, "
                         "or run_gcs_bridge.py's post-step modulo check misaligns "
                         "(architect review: the bug in the first attempted S5 fix)")

        # A later outage (the HTTP poll itself fails, not just one ally reporting dead)
        # must not un-latch `ready`, and -- this is what run_gcs_bridge.py actually reads
        # to decide whether to count a decision -- must not advance `_micro_ct` either.
        ally_link.raise_on_snapshot = True
        before = env._micro_ct
        env.step()
        self.assertTrue(env.ready, "ready is sticky once every hull has been seen once")
        self.assertEqual(env._micro_ct, before,
                         "a tick where the GCS poll itself fails must not advance "
                         "_micro_ct, even after quorum was already reached once")


if __name__ == "__main__":
    unittest.main()
