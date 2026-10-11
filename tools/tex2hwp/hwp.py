# -*- coding: utf-8 -*-
"""한컴오피스 한글 COM(HWPFrame.HwpObject) 얇은 래퍼. 문서 끝에 이어 쓰는 동작만 다룬다."""
from __future__ import annotations

import os

import win32com.client as win32

MM = 283.465  # 1 mm = 283.465 HWPUNIT


class Hwp:
    def __init__(self, visible: bool = False):
        self.h = win32.gencache.EnsureDispatch("HWPFrame.HwpObject")
        try:
            self.h.RegisterModule("FilePathCheckDLL", "FilePathCheckerModule")
        except Exception:
            pass
        self.h.XHwpWindows.Item(0).Visible = visible
        self.styles: dict[str, int] = {}

    # ── 파일 ────────────────────────────────────────────────────────────────
    def open(self, path: str, fmt: str = "") -> None:
        if not self.h.Open(os.path.abspath(path), fmt, "forceopen:true"):
            raise RuntimeError(f"open failed: {path}")

    def save_as(self, path: str, fmt: str) -> None:
        if not self.h.SaveAs(os.path.abspath(path), fmt, ""):
            raise RuntimeError(f"save failed: {path}")

    def quit(self) -> None:
        try:
            self.h.Clear(1)
        finally:
            self.h.Quit()

    # ── 커서 ────────────────────────────────────────────────────────────────
    def run(self, name: str) -> None:
        self.h.HAction.Run(name)

    def doc_end(self) -> None:
        self.run("MoveDocEnd")

    # ── 문단·글자 ───────────────────────────────────────────────────────────
    def style(self, index: int) -> None:
        act = self.h.CreateAction("Style")
        ps = act.CreateSet()
        act.GetDefault(ps)
        ps.SetItem("Apply", index)
        act.Execute(ps)

    def text(self, s: str) -> None:
        if not s:
            return
        act = self.h.CreateAction("InsertText")
        ps = act.CreateSet()
        act.GetDefault(ps)
        ps.SetItem("Text", s)
        act.Execute(ps)

    def para(self) -> None:
        self.run("BreakPara")

    def tab(self) -> None:
        self.run("InsertTab")

    def char_shape(self, **items) -> None:
        """현재 커서 이후 입력할 글자 모양. bold/italic/super/sub 는 0·1."""
        act = self.h.CreateAction("CharShape")
        ps = act.CreateSet()
        act.GetDefault(ps)
        key = {"bold": "Bold", "italic": "Italic", "super": "SuperScript", "sub": "SubScript",
               "height": "Height", "color": "TextColor"}
        for k, v in items.items():
            ps.SetItem(key[k], v)
        act.Execute(ps)

    def para_shape(self, **items) -> None:
        act = self.h.CreateAction("ParagraphShape")
        ps = act.CreateSet()
        act.GetDefault(ps)
        for k, v in items.items():
            ps.SetItem(k, v)
        act.Execute(ps)

    # ── 개체 ────────────────────────────────────────────────────────────────
    def equation(self, script: str, base_unit: int = 900) -> None:
        """글자처럼 취급되는 수식을 커서 위치에 넣고 커서를 그 뒤로 보낸다."""
        act = self.h.CreateAction("EquationCreate")
        ps = act.CreateSet()
        act.GetDefault(ps)
        ps.SetItem("String", script)
        ps.SetItem("BaseUnit", base_unit)
        ps.SetItem("Version", "Equation Version 60")
        act.Execute(ps)
        # 삽입 직후 개체가 선택된 상태일 수 있다 --- 선택을 풀고 개체 뒤로 이동
        self.run("Cancel")

    def picture(self, path: str, width_mm: float) -> None:
        from PIL import Image
        w, h = Image.open(path).size
        height_mm = width_mm * h / w
        ctrl = self.h.InsertPicture(os.path.abspath(path), True, 1, False, False, 0,
                                    int(round(width_mm)), int(round(height_mm)))
        if ctrl is not None:
            pset = ctrl.Properties
            pset.SetItem("TreatAsChar", True)
            pset.SetItem("Width", int(width_mm * MM))
            pset.SetItem("Height", int(height_mm * MM))
            ctrl.Properties = pset
        self.run("Cancel")
