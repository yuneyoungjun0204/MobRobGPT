"""위성사진 배경 타일 페치 (Esri World Imagery) → (이미지, meter-extent).

시뮬 좌표계(0~world_size m, 중심=world/2)에 맞춰, 지오 앵커(lat0,lon0) 주변
world_size×world_size 영역의 위성 타일을 받아 stitch → renderer.draw_scene(bg_img, bg_extent)로 전달.

- 인터넷 필요(첫 1회). 실패/오프라인 → None 반환 → 렌더러가 해색 배경으로 폴백.
- 캐시: 같은 (lat,lon,world,zoom) 은 메모리 캐시.
"""
from __future__ import annotations

import io
import math

_CACHE: dict = {}
_TILE_URL = ("https://services.arcgisonline.com/ArcGIS/rest/services/"
             "World_Imagery/MapServer/tile/{z}/{y}/{x}")


def _deg2num(lat: float, lon: float, z: int):
    lat_r = math.radians(lat)
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def _num2deg(x: float, y: float, z: int):
    n = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y / n))))
    return lat, lon


def fetch_satellite_bg(lat0: float, lon0: float, world_size: float,
                       zoom: int = 19, timeout: float = 6.0):
    """→ (numpy 이미지, [xmin,xmax,ymin,ymax] meters) 또는 실패 시 None."""
    key = (round(lat0, 6), round(lon0, 6), round(world_size, 3), zoom)
    if key in _CACHE:
        return _CACHE[key]
    try:
        import urllib.request
        import numpy as np
        from PIL import Image

        half = world_size / 2.0
        m_per_lat = 111320.0
        m_per_lon = 111320.0 * math.cos(math.radians(lat0))
        lat_n = lat0 + half / m_per_lat
        lat_s = lat0 - half / m_per_lat
        lon_w = lon0 - half / m_per_lon
        lon_e = lon0 + half / m_per_lon

        xw, yn = _deg2num(lat_n, lon_w, zoom)   # 북서
        xe, ys = _deg2num(lat_s, lon_e, zoom)   # 남동
        x0, x1 = int(math.floor(xw)), int(math.floor(xe))
        y0, y1 = int(math.floor(yn)), int(math.floor(ys))
        cols, rows = x1 - x0 + 1, y1 - y0 + 1
        if cols < 1 or rows < 1 or cols * rows > 25:
            return None

        mosaic = Image.new("RGB", (cols * 256, rows * 256))
        for xi in range(x0, x1 + 1):
            for yi in range(y0, y1 + 1):
                url = _TILE_URL.format(z=zoom, y=yi, x=xi)
                req = urllib.request.Request(url, headers={"User-Agent": "MobRobGPT/1.0"})
                data = urllib.request.urlopen(req, timeout=timeout).read()
                tile = Image.open(io.BytesIO(data)).convert("RGB")
                mosaic.paste(tile, ((xi - x0) * 256, (yi - y0) * 256))

        # 모자이크의 지리 경계 → meter extent (중심 기준, 0~world_size 프레임)
        latN, lonW = _num2deg(x0, y0, zoom)          # 좌상(북서) 코너
        latS, lonE = _num2deg(x1 + 1, y1 + 1, zoom)  # 우하(남동) 코너
        xmin = (lonW - lon0) * m_per_lon + half
        xmax = (lonE - lon0) * m_per_lon + half
        ymin = (latS - lat0) * m_per_lat + half
        ymax = (latN - lat0) * m_per_lat + half
        img = np.asarray(mosaic)                     # 행0=북(위) → origin="upper" 와 일치
        result = (img, [xmin, xmax, ymin, ymax])
        _CACHE[key] = result
        return result
    except Exception:
        return None


__all__ = ["fetch_satellite_bg"]
