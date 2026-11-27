"""Attachments, drawn over a viewport.

Select an attached object and its parent's face is tinted, the edges it is
measured from are drawn heavier, and a dimension runs from each edge to it
with its distance (or share); a badge names what it is attached to. Select
the parent and what follows it is outlined, so you see what will move with
it. What to draw comes from core.attachments.marks; this only paints it,
in screen space, over the model.
"""

from __future__ import annotations

import numpy as np

FACE = (90, 170, 255)
EDGE = (255, 170, 60)
TEXT = (235, 235, 235)


def paint(viewport, w, h):
    from ..core import attachments
    ids = list(getattr(viewport.selection, "ids", ()) or ())
    if not ids or viewport.space != "model":
        return
    try:
        marks = attachments.marks(viewport.scene, ids)
    except Exception:                                       # noqa: BLE001
        return
    if not any(marks.values()):
        return
    from OpenGL import GL
    from PySide6.QtCore import QPointF, QRectF
    from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF

    def screen(points):
        pts = np.asarray(points, float).reshape(-1, 3)
        scr = viewport.camera.project(pts, w, h)
        return [None if s[2] <= 0 else QPointF(float(s[0]), float(s[1]))
                for s in scr]

    GL.glBindVertexArray(0)
    viewport._use(0)
    GL.glDisable(GL.GL_DEPTH_TEST)
    painter = QPainter(viewport)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    try:
        painter.setPen(QPen(QColor(*FACE, 200), 1.5))
        painter.setBrush(QColor(*FACE, 50))
        for loop in marks["faces"]:
            pts = screen(loop)
            if all(p is not None for p in pts):
                painter.drawPolygon(QPolygonF(pts))
        painter.setBrush(QColor(0, 0, 0, 0))
        painter.setPen(QPen(QColor(*FACE, 150), 1.2))
        for lines in marks["followers"]:
            for line in lines:
                pts = screen(line)
                for a, b in zip(pts, pts[1:]):
                    if a is not None and b is not None:
                        painter.drawLine(a, b)
        painter.setPen(QPen(QColor(*EDGE, 230), 3.0))
        for p, q in marks["edges"]:
            a, b = screen([p, q])
            if a is not None and b is not None:
                painter.drawLine(a, b)
        font = QFont(viewport.font())
        font.setPointSizeF(9.0)
        painter.setFont(font)
        fm = painter.fontMetrics()

        def bubble(at, text, border):
            tw, th, pad = fm.horizontalAdvance(text), fm.height(), 5.0
            rect = QRectF(at.x() - tw / 2 - pad, at.y() - th / 2 - 3,
                          tw + 2 * pad, th + 6)
            painter.setPen(QPen(QColor(*border), 1.2))
            painter.setBrush(QColor(38, 38, 42, 225))
            painter.drawRoundedRect(rect, 4, 4)
            painter.setPen(QColor(*TEXT))
            painter.drawText(rect, 0x84, text)

        for foot, p, text in marks["dims"]:
            a, b = screen([foot, p])
            if a is None or b is None:
                continue
            painter.setPen(QPen(QColor(*EDGE, 230), 1.4))
            painter.drawLine(a, b)
            for end in (a, b):
                painter.drawEllipse(end, 2.5, 2.5)
            bubble(QPointF((a.x() + b.x()) / 2, (a.y() + b.y()) / 2), text,
                   EDGE)
        for at, text in marks["badges"]:
            (s,) = screen([at])
            if s is not None:
                bubble(QPointF(s.x(), s.y() - 14), text, FACE)
    finally:
        painter.end()
        GL.glEnable(GL.GL_DEPTH_TEST)
