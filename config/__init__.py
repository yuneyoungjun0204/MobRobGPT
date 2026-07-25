"""공통 설정 로더.

defense_config.json을 읽어서 모든 컴포넌트가 동일한 설정을 사용하도록 합니다.
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def get_config_path() -> Path:
    """설정 파일 경로 반환."""
    if env_path := os.environ.get("DEFENSE_CONFIG"):
        return Path(env_path)
    return Path(__file__).parent / "defense_config.json"


def load_config(force_reload: bool = False) -> Dict[str, Any]:
    """설정 파일 로드 (캐싱)."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE
    config_path = get_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"설정 파일 없음: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        _CONFIG_CACHE = json.load(f)
    return _CONFIG_CACHE


def get_world_size() -> float:
    """월드 크기 (meters)."""
    return float(load_config()["world"]["size"])


def get_gps_origin() -> Tuple[float, float]:
    """GPS 원점 (lat, lon)."""
    origin = load_config()["gps_origin"]
    return float(origin["lat"]), float(origin["lon"])


def get_ship_counts() -> Tuple[int, int]:
    """선박 수 (n_allies, n_enemies)."""
    ships = load_config()["ships"]
    return int(ships["n_allies"]), int(ships["n_enemies"])
