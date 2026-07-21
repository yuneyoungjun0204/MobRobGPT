"""
commander/geo_bridge.py — GPS ↔ ENU ↔ 시뮬 좌표 변환 브리지

핵심 불변식:
  1. t=0에 affine 변환을 1회 결정하고 에피소드 동안 고정
  2. 좌표계 단일화: 내부는 전부 시뮬 좌표 (nav: x=East, y=North)
  3. 변환은 입력 경계(GPS→sim)와 출력 경계(sim→GPS) 두 곳에서만

좌표계:
  - GPS: 위경도 (lat, lon) in degrees
  - ENU: 지역 평면 (East, North, Up) in meters
  - SIM: 시뮬레이터 좌표 (x=East, y=North) in meters, 맵 [0, world_size]²
  - Heading: nav 규약 (0°=North, CW+)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional


@dataclass
class GeoBridge:
    """GPS ↔ ENU ↔ 시뮬 좌표 변환.

    사용법:
        bridge = GeoBridge(world_size=12600, target_sim_radius=5450)
        bridge.fit(allies_geo, enemies_geo)  # t=0에 1회
        sim_xy = bridge.to_sim(gps_lat, gps_lon)
        gps_lat, gps_lon = bridge.to_geo(sim_xy)
    """

    # 시뮬 설정
    world_size: float = 12600.0
    target_sim_radius: float = 5450.0  # 적을 이 반경에 매핑

    # 스케일 범위 (과확대/과축소 방지)
    scale_min: float = 0.5
    scale_max: float = 2.0
    d_min_floor: float = 1000.0  # 적 거리 최소값 (과확대 방지)

    # 변환 파라미터 (fit 후 결정)
    origin_lat: float = field(default=0.0, init=False)
    origin_lon: float = field(default=0.0, init=False)
    center_enu: np.ndarray = field(default_factory=lambda: np.zeros(2), init=False)
    scale: float = field(default=1.0, init=False)
    sim_center: np.ndarray = field(default_factory=lambda: np.zeros(2), init=False)
    _fitted: bool = field(default=False, init=False)

    def __post_init__(self):
        self.sim_center = np.array([self.world_size / 2, self.world_size / 2])

    def _gps_to_enu(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """GPS(위경도) → ENU(미터). 등거리 근사 (소규모 해역).

        Args:
            lat, lon: [...] shape 배열

        Returns:
            enu: [..., 2] (East, North) in meters
        """
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)

        d_north = (lat - self.origin_lat) * 111320.0
        d_east = (lon - self.origin_lon) * 111320.0 * np.cos(np.deg2rad(self.origin_lat))

        return np.stack([d_east, d_north], axis=-1)

    def _enu_to_gps(self, enu: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ENU(미터) → GPS(위경도). _gps_to_enu의 역함수.

        Args:
            enu: [..., 2] (East, North) in meters

        Returns:
            (lat, lon): [...] shape 배열
        """
        enu = np.asarray(enu, dtype=np.float64)
        d_east = enu[..., 0]
        d_north = enu[..., 1]

        lat = self.origin_lat + d_north / 111320.0
        lon = self.origin_lon + d_east / (111320.0 * np.cos(np.deg2rad(self.origin_lat)))

        return lat, lon

    def fit(self, allies_geo: np.ndarray, enemies_geo: np.ndarray,
            mothership_geo: Optional[Tuple[float, float]] = None) -> "GeoBridge":
        """t=0에 affine 변환 파라미터 결정.

        Args:
            allies_geo: [P, 2] 아군 (lat, lon)
            enemies_geo: [M, 2] 적 (lat, lon)
            mothership_geo: (lat, lon) 모선. None이면 아군 무게중심 사용.

        Returns:
            self (체이닝용)
        """
        allies_geo = np.asarray(allies_geo, dtype=np.float64)
        enemies_geo = np.asarray(enemies_geo, dtype=np.float64)

        # 기준점 (origin_geo): 모선 또는 아군 무게중심
        if mothership_geo is not None:
            self.origin_lat, self.origin_lon = mothership_geo
        else:
            self.origin_lat = allies_geo[:, 0].mean()
            self.origin_lon = allies_geo[:, 1].mean()

        # 방어 중심 (ENU): 모선 위치 = ENU 원점 (0, 0)
        self.center_enu = np.array([0.0, 0.0])

        # 적들의 ENU 좌표
        if len(enemies_geo) > 0:
            enemies_enu = self._gps_to_enu(enemies_geo[:, 0], enemies_geo[:, 1])
            # 적의 최대 거리
            d_max = np.max(np.linalg.norm(enemies_enu - self.center_enu, axis=-1))
            d_max = max(d_max, self.d_min_floor)
        else:
            d_max = self.d_min_floor

        # 스케일 결정: 적을 target_sim_radius에 매핑
        self.scale = self.target_sim_radius / d_max
        self.scale = np.clip(self.scale, self.scale_min, self.scale_max)

        self._fitted = True
        return self

    def to_sim(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """GPS → 시뮬 좌표.

        Args:
            lat, lon: [...] shape 배열

        Returns:
            sim_xy: [..., 2] in [0, world_size]
        """
        if not self._fitted:
            raise RuntimeError("fit()을 먼저 호출하세요")

        enu = self._gps_to_enu(lat, lon)
        sim_xy = self.sim_center + (enu - self.center_enu) * self.scale
        return np.clip(sim_xy, 0.0, self.world_size)

    def to_geo(self, sim_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """시뮬 좌표 → GPS.

        Args:
            sim_xy: [..., 2]

        Returns:
            (lat, lon): [...] shape 배열
        """
        if not self._fitted:
            raise RuntimeError("fit()을 먼저 호출하세요")

        sim_xy = np.asarray(sim_xy, dtype=np.float64)
        enu = self.center_enu + (sim_xy - self.sim_center) / self.scale
        return self._enu_to_gps(enu)

    def hdg_to_sim(self, yaw_enu_deg: np.ndarray) -> np.ndarray:
        """ENU yaw → 시뮬 heading.

        ENU yaw: 0=East, CCW+
        SIM heading: 0=North, CW+ (nav 규약)

        Args:
            yaw_enu_deg: [...] ENU yaw in degrees

        Returns:
            hdg_sim: [...] nav heading in degrees [0, 360)
        """
        yaw_enu_deg = np.asarray(yaw_enu_deg, dtype=np.float64)
        hdg_sim = (90.0 - yaw_enu_deg) % 360.0
        return hdg_sim

    def hdg_to_enu(self, hdg_sim_deg: np.ndarray) -> np.ndarray:
        """시뮬 heading → ENU yaw. hdg_to_sim의 역함수.

        Args:
            hdg_sim_deg: [...] nav heading in degrees

        Returns:
            yaw_enu: [...] ENU yaw in degrees
        """
        hdg_sim_deg = np.asarray(hdg_sim_deg, dtype=np.float64)
        yaw_enu = (90.0 - hdg_sim_deg) % 360.0
        return yaw_enu

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def get_transform_info(self) -> dict:
        """변환 파라미터 조회 (디버깅용)."""
        return {
            "origin_lat": self.origin_lat,
            "origin_lon": self.origin_lon,
            "center_enu": self.center_enu.tolist(),
            "scale": self.scale,
            "sim_center": self.sim_center.tolist(),
            "world_size": self.world_size,
            "target_sim_radius": self.target_sim_radius,
        }
