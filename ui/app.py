
"""
LP Solver — PySide6 GUI
Supports: Standard Simplex | Two-Phase Simplex
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QComboBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QScrollArea, QFrame,
    QSplitter, QGroupBox, QButtonGroup, QRadioButton,
    QHeaderView, QSizePolicy, QMessageBox, QStackedWidget,
    QGridLayout, QAbstractItemView, QSizePolicy
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QColor

from engine.models import LPProblem
from engine.standard import StandardSimplex
from engine.two_phase import TwoPhaseSimplex


# ══════════════════════════════════════════════════════════
#  STYLESHEET
# ══════════════════════════════════════════════════════════
STYLE = """
QWidget {
    background-color: #F8F9FB;
    color: #1C1F26;
    font-family: "Segoe UI", "SF Pro Text", Arial, sans-serif;
    font-size: 13px;
}
/* ── Header ── */
#header {
    background-color: #1B3358;
    min-height: 58px; max-height: 58px;
}
#appTitle {
    font-size: 17px; font-weight: 700;
    color: #FFFFFF; letter-spacing: 0.4px;
}
#appSub {
    font-size: 11px; color: #90B8E0; letter-spacing: 0.3px;
}
/* ── Sidebar ── */
#sidebar {
    background-color: #FFFFFF;
    border-right: 1px solid #E3E7EF;
    min-width: 270px; max-width: 270px;
}
#sideTitle {
    font-size: 15px; font-weight: 700;
    color: #1B3358; margin-bottom: 2px;
}
#secLabel {
    font-size: 10px; font-weight: 700;
    letter-spacing: 1.3px; color: #8A93A6;
    text-transform: uppercase;
}
/* ── Buttons ── */
#solveBtn {
    background-color: #1D56C4;
    color: #FFFFFF; border: none;
    border-radius: 8px; font-size: 14px;
    font-weight: 600; padding: 11px 26px;
}
#solveBtn:hover  { background-color: #174DB0; }
#solveBtn:pressed { background-color: #133E96; }
#solveBtn:disabled { background-color: #7FAEE8; }
#resetBtn {
    background-color: transparent;
    color: #6B7280; border: 1px solid #D1D5DB;
    border-radius: 8px; font-size: 13px;
    padding: 9px 18px;
}
#resetBtn:hover { background-color: #F3F4F6; }
/* ── Radio ── */
QRadioButton { spacing: 7px; font-size: 13px; color: #374151; }
QRadioButton::indicator {
    width: 15px; height: 15px;
    border-radius: 8px; border: 2px solid #C5CAD4;
    background: white;
}
QRadioButton::indicator:checked {
    border: 2px solid #1D56C4; background-color: #1D56C4;
}
/* ── SpinBox ── */
QSpinBox {
    background: #FFFFFF; border: 1px solid #D1D5DB;
    border-radius: 6px; padding: 5px 7px;
    font-size: 13px; color: #111827; min-width: 65px;
}
QSpinBox:focus { border: 1.5px solid #1D56C4; }
QSpinBox::up-button, QSpinBox::down-button { width: 18px; border: none; background: transparent; }
/* ── ComboBox ── */
QComboBox {
    background: #FFFFFF; border: 1px solid #D1D5DB;
    border-radius: 6px; padding: 5px 7px;
    font-size: 13px; color: #111827;
}
QComboBox:focus { border: 1.5px solid #1D56C4; }
QComboBox::drop-down { border: none; width: 22px; }
QComboBox QAbstractItemView {
    background: #FFFFFF; border: 1px solid #D1D5DB;
    border-radius: 4px;
    selection-background-color: #EEF4FF;
    selection-color: #1D56C4;
}
/* ── Input tables ── */
#inputTable {
    background: #FFFFFF; border: 1px solid #E3E7EF;
    border-radius: 6px; gridline-color: #E3E7EF;
}
#inputTable QHeaderView::section {
    background: #F0F4FA; color: #4B5563;
    font-weight: 600; font-size: 11px;
    border: none;
    border-right: 1px solid #E3E7EF;
    border-bottom: 1px solid #E3E7EF;
    padding: 5px 8px;
}
#inputTable::item { padding: 3px 6px; }
#inputTable::item:selected { background: #EEF4FF; color: #1D56C4; }
/* ── Result tableau ── */
#tableau {
    background: #FFFFFF; border: 1px solid #E3E7EF;
    border-radius: 6px; gridline-color: #E9ECF2;
    font-size: 12px;
    font-family: "JetBrains Mono","Consolas","Courier New",monospace;
}
#tableau QHeaderView::section:horizontal {
    background: #1B3358; color: #FFFFFF;
    font-weight: 600; font-size: 11px;
    font-family: "Segoe UI", Arial, sans-serif;
    border: none;
    border-right: 1px solid #264D7A;
    border-bottom: 1px solid #264D7A;
    padding: 6px 10px;
}
#tableau QHeaderView::section:vertical {
    background: #EEF4FF; color: #1D56C4;
    font-weight: 600; font-size: 11px;
    font-family: "Segoe UI", Arial, sans-serif;
    border: none;
    border-bottom: 1px solid #DBEAFE;
    border-right: 1px solid #DBEAFE;
    padding: 4px 8px;
}
#tableau::item { padding: 4px 10px; color: #1F2937; }
#tableau::item:selected { background: #EEF4FF; }
/* ── Tabs ── */
QTabWidget::pane {
    border: 1px solid #E3E7EF; border-radius: 8px;
    background: #FFFFFF; top: -1px;
}
QTabBar::tab {
    background: #F0F4FA; color: #6B7280;
    border: 1px solid #E3E7EF; border-bottom: none;
    border-radius: 6px 6px 0 0;
    padding: 6px 14px; margin-right: 3px; font-size: 12px;
}
QTabBar::tab:selected {
    background: #FFFFFF; color: #1D56C4;
    font-weight: 700; border-bottom: 2px solid #1D56C4;
}
QTabBar::tab:hover:!selected { background: #E5EDF7; }
/* Phase badge on tab */
QTabBar::tab[phase="1"] { border-top: 3px solid #F59E0B; }
QTabBar::tab[phase="2"] { border-top: 3px solid #10B981; }
/* ── Status badges ── */
#badgeOptimal    { background:#D1FAE5; color:#065F46; border-radius:12px; padding:4px 14px; font-weight:700; }
#badgeInfeasible { background:#FEE2E2; color:#991B1B; border-radius:12px; padding:4px 14px; font-weight:700; }
#badgeUnbounded  { background:#FEF3C7; color:#92400E; border-radius:12px; padding:4px 14px; font-weight:700; }
/* ── Result card ── */
#card { background:#FFFFFF; border:1px solid #E3E7EF; border-radius:10px; padding:14px; }
#zValue { font-size:30px; font-weight:700; color:#1D56C4; }
#zLabel { font-size:10px; font-weight:700; color:#9CA3AF; letter-spacing:1px; }
#varVal { font-size:14px; font-weight:700; color:#1F2937; }
#varLbl { font-size:11px; color:#6B7280; }
#varChip {
    background:#F0F9FF; border:1px solid #BAE6FD;
    border-radius:8px; padding:6px 14px;
}
/* ── GroupBox ── */
QGroupBox {
    border:1px solid #E3E7EF; border-radius:8px;
    margin-top:10px; padding:8px; font-size:12px;
    font-weight:600; color:#374151;
}
QGroupBox::title { subcontrol-origin:margin; left:10px; padding:0 4px; }
/* ── Divider ── */
#divider { background:#E3E7EF; max-height:1px; }
/* ── Scrollbars ── */
QScrollBar:vertical   { background:#F1F5F9; width:7px; border-radius:4px; }
QScrollBar:horizontal { background:#F1F5F9; height:7px; border-radius:4px; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal
    { background:#CBD5E1; border-radius:4px; min-height:20px; min-width:20px; }
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background:#94A3B8; }
QScrollBar::add-line, QScrollBar::sub-line { width:0; height:0; }
/* ── Phase banner ── */
#phase1Banner { background:#FEF3C7; border-radius:6px; padding:4px 12px; }
#phase2Banner { background:#D1FAE5; border-radius:6px; padding:4px 12px; }
#phaseLbl1 { font-size:11px; font-weight:700; color:#92400E; }
#phaseLbl2 { font-size:11px; font-weight:700; color:#065F46; }
"""


# ══════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════
def sec_label(text: str) -> QLabel:
    lbl = QLabel(text.upper())
    lbl.setObjectName("secLabel")
    return lbl

def divider() -> QFrame:
    f = QFrame(); f.setObjectName("divider")
    f.setFrameShape(QFrame.HLine); f.setFixedHeight(1)
    return f


# ══════════════════════════════════════════════════════════
#  SOLVER THREAD
# ══════════════════════════════════════════════════════════
class SolverThread(QThread):
    done  = Signal(dict)
    error = Signal(str)

    def __init__(self, problem: LPProblem, method: str):
        super().__init__()
        self.problem = problem
        self.method = method

    def run(self):
        try:
            solver = StandardSimplex(self.problem) if self.method == 'standard' \
                     else TwoPhaseSimplex(self.problem)
            self.done.emit(solver.solve())
        except Exception as e:
            self.error.emit(str(e))


# ══════════════════════════════════════════════════════════
#  MATRIX INPUT TABLE
# ══════════════════════════════════════════════════════════
class MatrixTable(QTableWidget):
    def __init__(self, rows: int, cols: int):
        super().__init__(rows, cols)
        self.setObjectName("inputTable")
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.verticalHeader().setDefaultSectionSize(32)
        self.setAlternatingRowColors(False)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self._fill_zeros()

    def _fill_zeros(self):
        for r in range(self.rowCount()):
            for c in range(self.columnCount()):
                self._set(r, c, "0")

    def _set(self, r, c, text="0"):
        item = QTableWidgetItem(text)
        item.setTextAlignment(Qt.AlignCenter)
        self.setItem(r, c, item)

    def resize_to(self, rows: int, cols: int):
        self.setRowCount(rows); self.setColumnCount(cols)
        for r in range(rows):
            for c in range(cols):
                if self.item(r, c) is None:
                    self._set(r, c)

    def get_row(self, r: int) -> list[float]:
        out = []
        for c in range(self.columnCount()):
            try:    out.append(float(self.item(r, c).text()))
            except: out.append(0.0)
        return out

    def get_all(self) -> list[list[float]]:
        return [self.get_row(r) for r in range(self.rowCount())]


# ══════════════════════════════════════════════════════════
#  TABLEAU DISPLAY WIDGET
# ══════════════════════════════════════════════════════════
class TableauView(QTableWidget):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.setObjectName("tableau")
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.setAlternatingRowColors(True)
        self._load(df)

    def _load(self, df: pd.DataFrame):
        self.setRowCount(df.shape[0])
        self.setColumnCount(df.shape[1])
        self.setHorizontalHeaderLabels(list(map(str, df.columns)))
        self.setVerticalHeaderLabels(list(map(str, df.index)))
        n_rows, n_cols = df.shape
        for r in range(n_rows):
            for c in range(n_cols):
                val = df.iloc[r, c]
                item = QTableWidgetItem(f"{val:.4f}")
                item.setTextAlignment(Qt.AlignCenter)
                is_obj_row = (r == n_rows - 1)
                is_rhs_col = (c == n_cols - 1)
                # Reset colors first
                item.setForeground(QColor("#1F2937"))
                item.setBackground(QColor("#FFFFFF"))
                if is_obj_row:
                    item.setBackground(QColor("#EEF4FF"))
                    item.setForeground(QColor("#1E40AF"))
                if is_rhs_col:
                    item.setForeground(QColor("#059669"))
                    f = item.font();
                    f.setBold(True);
                    item.setFont(f)
                self.setItem(r, c, item)


# ══════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════
class Sidebar(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("sidebar")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(14)

        # Title
        t = QLabel("Configuration"); t.setObjectName("sideTitle")
        lay.addWidget(t)
        lay.addWidget(divider())

        # Objective
        lay.addWidget(sec_label("Objective"))
        self.max_r = QRadioButton("Maximize"); self.max_r.setChecked(True)
        self.min_r = QRadioButton("Minimize")
        bg = QButtonGroup(self); bg.addButton(self.max_r); bg.addButton(self.min_r)
        row = QHBoxLayout(); row.addWidget(self.max_r); row.addWidget(self.min_r); row.addStretch()
        lay.addLayout(row)

        # Method
        lay.addWidget(divider())
        lay.addWidget(sec_label("Method"))
        self.std_r = QRadioButton("Standard Simplex"); self.std_r.setChecked(True)
        self.tp_r  = QRadioButton("Two-Phase")
        bg2 = QButtonGroup(self); bg2.addButton(self.std_r); bg2.addButton(self.tp_r)
        row2 = QHBoxLayout(); row2.addWidget(self.std_r); row2.addWidget(self.tp_r); row2.addStretch()
        lay.addLayout(row2)

        # Dimensions
        lay.addWidget(divider())
        lay.addWidget(sec_label("Dimensions"))
        g = QGridLayout(); g.setSpacing(8)
        g.addWidget(QLabel("Variables:"), 0, 0)
        self.var_sp = QSpinBox(); self.var_sp.setRange(1, 10); self.var_sp.setValue(2)
        g.addWidget(self.var_sp, 0, 1)
        g.addWidget(QLabel("Constraints:"), 1, 0)
        self.con_sp = QSpinBox(); self.con_sp.setRange(1, 10); self.con_sp.setValue(3)
        g.addWidget(self.con_sp, 1, 1)
        lay.addLayout(g)

        self.var_sp.valueChanged.connect(self.changed)
        self.con_sp.valueChanged.connect(self.changed)

        # Hint label for two-phase
        self.hint = QLabel("💡 Two-Phase handles ≥ and = constraints.")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("font-size:11px; color:#6B7280; margin-top:4px;")
        self.hint.setVisible(False)
        self.tp_r.toggled.connect(self.hint.setVisible)
        lay.addWidget(self.hint)

        lay.addStretch()

    @property
    def n_vars(self): return self.var_sp.value()
    @property
    def n_cons(self): return self.con_sp.value()
    @property
    def is_max(self): return self.max_r.isChecked()
    @property
    def method(self): return 'standard' if self.std_r.isChecked() else 'two_phase'


# ══════════════════════════════════════════════════════════
#  PROBLEM INPUT PANEL
# ══════════════════════════════════════════════════════════
class InputPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.n_vars = 2
        self.n_cons = 3
        self._type_combos: list[QComboBox] = []
        self._rest_combos: list[QComboBox] = []
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        body = QWidget()
        self._lay = QVBoxLayout(body)
        self._lay.setContentsMargins(20, 16, 20, 16)
        self._lay.setSpacing(14)

        # ── Objective ──
        grp_obj = QGroupBox("Objective Function  —  Coefficients cᵢ")
        gl = QVBoxLayout(grp_obj)
        self._obj = MatrixTable(1, self.n_vars)
        self._obj.setVerticalHeaderLabels(["c"])
        self._obj.setHorizontalHeaderLabels([f"x{i+1}" for i in range(self.n_vars)])
        self._obj.setFixedHeight(58)
        gl.addWidget(self._obj)
        self._lay.addWidget(grp_obj)

        # ── Constraints ──
        grp_con = QGroupBox("Constraint Matrix  [A | type | b]")
        gcl = QVBoxLayout(grp_con)
        self._A = MatrixTable(self.n_cons, self.n_vars)
        self._A.setVerticalHeaderLabels([f"C{i+1}" for i in range(self.n_cons)])
        self._A.setHorizontalHeaderLabels([f"x{i+1}" for i in range(self.n_vars)])
        gcl.addWidget(self._A)

        meta = QHBoxLayout(); meta.setSpacing(10)

        # Types
        grp_type = QGroupBox("Type")
        self._type_lay = QVBoxLayout(grp_type)
        self._type_lay.setSpacing(4)
        self._type_combos = []
        for _ in range(self.n_cons):
            cb = QComboBox(); cb.addItems(["<=", ">=", "="]); cb.setFixedWidth(75)
            self._type_lay.addWidget(cb); self._type_combos.append(cb)
        meta.addWidget(grp_type)

        # RHS
        grp_rhs = QGroupBox("RHS  (b)")
        rhs_l = QVBoxLayout(grp_rhs)
        self._rhs = MatrixTable(self.n_cons, 1)
        self._rhs.setHorizontalHeaderLabels(["b"])
        rhs_l.addWidget(self._rhs)
        meta.addWidget(grp_rhs, 1)

        gcl.addLayout(meta)
        self._lay.addWidget(grp_con)

        # ── Restrictions ──
        grp_rest = QGroupBox("Variable Restrictions")
        self._rest_lay = QHBoxLayout(grp_rest)
        self._rest_lay.setSpacing(10)
        self._rest_combos = []
        for i in range(self.n_vars):
            col = QVBoxLayout()
            lbl = QLabel(f"x{i+1}"); lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight:600; font-size:11px; color:#374151;")
            cb = QComboBox(); cb.addItems(["≥ 0", "Unrestricted"])
            col.addWidget(lbl); col.addWidget(cb)
            w = QWidget(); w.setLayout(col)
            self._rest_lay.addWidget(w)
            self._rest_combos.append(cb)
        self._rest_lay.addStretch()
        self._lay.addWidget(grp_rest)

        self._lay.addStretch()
        scroll.setWidget(body)
        outer.addWidget(scroll)

    # ── Rebuild on dimension change ──
    def rebuild(self, n_vars: int, n_cons: int):
        self.n_vars = n_vars
        self.n_cons = n_cons

        # Objective
        self._obj.resize_to(1, n_vars)
        self._obj.setHorizontalHeaderLabels([f"x{i+1}" for i in range(n_vars)])

        # A matrix
        self._A.resize_to(n_cons, n_vars)
        self._A.setHorizontalHeaderLabels([f"x{i+1}" for i in range(n_vars)])
        self._A.setVerticalHeaderLabels([f"C{i+1}" for i in range(n_cons)])

        # RHS
        self._rhs.resize_to(n_cons, 1)

        # Type combos
        for cb in self._type_combos: cb.setParent(None); cb.deleteLater()
        self._type_combos.clear()
        while self._type_lay.count():
            self._type_lay.takeAt(0)
        for _ in range(n_cons):
            cb = QComboBox(); cb.addItems(["<=", ">=", "="]); cb.setFixedWidth(75)
            self._type_lay.addWidget(cb); self._type_combos.append(cb)

        # Restriction combos
        for cb in self._rest_combos: cb.setParent(None); cb.deleteLater()
        self._rest_combos.clear()
        # remove all items from rest layout
        while self._rest_lay.count():
            item = self._rest_lay.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for i in range(n_vars):
            col = QVBoxLayout()
            lbl = QLabel(f"x{i+1}"); lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("font-weight:600; font-size:11px; color:#374151;")
            cb = QComboBox(); cb.addItems(["≥ 0", "Unrestricted"])
            col.addWidget(lbl); col.addWidget(cb)
            w = QWidget(); w.setLayout(col)
            self._rest_lay.addWidget(w)
            self._rest_combos.append(cb)
        self._rest_lay.addStretch()

    # ── Extract data ──
    def get_data(self) -> dict:
        return {
            'c':    self._obj.get_row(0),
            'A':    self._A.get_all(),
            'b':    [self._rhs.get_row(r)[0] for r in range(self.n_cons)],
            'types': [cb.currentText() for cb in self._type_combos] or ["<="] * self.n_cons,
            'restrictions': [
                ">=0" if cb.currentIndex() == 0 else "unrestricted"
                for cb in self._rest_combos
            ] or [">=0"] * self.n_vars,
        }


# ══════════════════════════════════════════════════════════
#  RESULT PANEL
# ══════════════════════════════════════════════════════════
class ResultPanel(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 14, 20, 14)
        lay.setSpacing(14)

        self.stack = QStackedWidget()

        # ─ Empty page ─
        empty = QWidget()
        el = QVBoxLayout(empty); el.addStretch()
        msg = QLabel("Configure a problem on the left\nand press  ▶  Solve.")
        msg.setAlignment(Qt.AlignCenter)
        msg.setStyleSheet("color:#9CA3AF; font-size:14px; line-height:1.6;")
        el.addWidget(msg); el.addStretch()
        self.stack.addWidget(empty)          # index 0

        # ─ Result page ─
        res = QWidget()
        rl = QVBoxLayout(res); rl.setSpacing(14)

        # Summary card
        card = QFrame(); card.setObjectName("card")
        cl = QHBoxLayout(card); cl.setSpacing(28)

        sc = QVBoxLayout()
        sc.addWidget(QLabel("STATUS", objectName="zLabel"))
        self._status_lbl = QLabel("—"); self._status_lbl.setObjectName("badgeOptimal")
        sc.addWidget(self._status_lbl)
        cl.addLayout(sc)

        zc = QVBoxLayout()
        zc.addWidget(QLabel("OBJECTIVE VALUE  Z", objectName="zLabel"))
        self._z_lbl = QLabel("—"); self._z_lbl.setObjectName("zValue")
        zc.addWidget(self._z_lbl)
        cl.addLayout(zc)
        cl.addStretch()
        rl.addWidget(card)

        # Variables row
        var_frame = QFrame(); var_frame.setObjectName("card")
        self._var_lay = QHBoxLayout(var_frame); self._var_lay.setSpacing(10)
        rl.addWidget(var_frame)

        # Iterations
        iter_hdr = QHBoxLayout()
        iter_hdr.addWidget(sec_label("Simplex Iterations"))
        self._phase_legend = QLabel("")
        self._phase_legend.setStyleSheet("font-size:11px; color:#6B7280;")
        iter_hdr.addStretch()
        iter_hdr.addWidget(self._phase_legend)
        rl.addLayout(iter_hdr)

        self._tabs = QTabWidget()
        self._tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        rl.addWidget(self._tabs)

        self.stack.addWidget(res)            # index 1
        lay.addWidget(self.stack)

    # ── Public ──
    def show_empty(self):
        self.stack.setCurrentIndex(0)

    def show_result(self, result: dict):
        status  = result.get('status', 'unknown')
        z       = result.get('z')
        sol     = result.get('solution') or {}
        history = result.get('history') or []
        labels  = result.get('phase_labels') or [''] * len(history)

        # Filter out internal variables (surplus e, artificial a, slack s)
        display_sol = {k: v for k, v in sol.items()
                       if not (k.startswith('e') or k.startswith('a') or k.startswith('s'))}

        # Status badge
        badge_map = {
            'optimal':    'badgeOptimal',
            'infeasible': 'badgeInfeasible',
            'unbounded':  'badgeUnbounded',
        }
        self._status_lbl.setText(status.upper())
        self._status_lbl.setObjectName(badge_map.get(status, 'badgeOptimal'))
        self._status_lbl.style().unpolish(self._status_lbl)
        self._status_lbl.style().polish(self._status_lbl)

        # Z value
        self._z_lbl.setText(f"{z:.6g}" if z is not None else "N / A")

        # Variable chips
        while self._var_lay.count():
            item = self._var_lay.takeAt(0)
            if item.widget(): item.widget().deleteLater()

        for name, val in display_sol.items():
            chip = QFrame(); chip.setObjectName("varChip")
            cl = QVBoxLayout(chip); cl.setContentsMargins(10, 6, 10, 6); cl.setSpacing(2)
            nl = QLabel(name); nl.setObjectName("varLbl"); nl.setAlignment(Qt.AlignCenter)
            vl = QLabel(f"{float(val):.4f}"); vl.setObjectName("varVal"); vl.setAlignment(Qt.AlignCenter)
            cl.addWidget(nl); cl.addWidget(vl)
            self._var_lay.addWidget(chip)
        self._var_lay.addStretch()

        # Iteration tabs
        self._tabs.clear()
        has_phases = any(lbl for lbl in labels)
        self._phase_legend.setText(
            "🟡 Phase 1  |  🟢 Phase 2" if has_phases else ""
        )

        for i, df in enumerate(history):
            tab = QWidget()
            tl = QVBoxLayout(tab); tl.setContentsMargins(8, 8, 8, 8)
            tl.addWidget(TableauView(df))
            self._tabs.addTab(tab, f"Iter {i}" if i > 0 else "Initial")

            # Color-code tab by phase
            if has_phases and i < len(labels):
                ph = labels[i]
                color = "#F59E0B" if ph == "Phase 1" else "#10B981"
                self._tabs.tabBar().setTabTextColor(i, QColor(color) if ph else QColor("#6B7280"))

        self.stack.setCurrentIndex(1)


# ══════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("LP Solver — Simplex Method")
        self.resize(1300, 840)
        self._worker = None
        self._build()

    def _build(self):
        root = QWidget(); self.setCentralWidget(root)
        vl = QVBoxLayout(root); vl.setContentsMargins(0,0,0,0); vl.setSpacing(0)

        # ── Header ──
        hdr = QFrame(); hdr.setObjectName("header"); hdr.setFixedHeight(58)
        hl = QHBoxLayout(hdr); hl.setContentsMargins(20,0,20,0)
        tc = QVBoxLayout(); tc.setSpacing(1)
        tc.addWidget(QLabel("LP Solver", objectName="appTitle"))
        tc.addWidget(QLabel("Standard Simplex  &  Two-Phase Method", objectName="appSub"))
        hl.addLayout(tc); hl.addStretch()

        self._solve_btn = QPushButton("▶  Solve")
        self._solve_btn.setObjectName("solveBtn")
        self._solve_btn.setFixedHeight(36)
        self._solve_btn.setCursor(Qt.PointingHandCursor)
        self._solve_btn.clicked.connect(self._solve)

        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setObjectName("resetBtn")
        self._reset_btn.setFixedHeight(36)
        self._reset_btn.setCursor(Qt.PointingHandCursor)
        self._reset_btn.clicked.connect(self._reset)

        hl.addWidget(self._reset_btn); hl.addSpacing(8); hl.addWidget(self._solve_btn)
        vl.addWidget(hdr)

        # ── Body ──
        body = QWidget()
        bl = QHBoxLayout(body); bl.setContentsMargins(0,0,0,0); bl.setSpacing(0)

        self._sidebar = Sidebar()
        self._sidebar.changed.connect(self._on_dim_change)
        bl.addWidget(self._sidebar)

        sep = QFrame(); sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("background:#E3E7EF; max-width:1px;")
        bl.addWidget(sep)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background:#E3E7EF; }")

        self._input  = InputPanel()
        self._result = ResultPanel()
        splitter.addWidget(self._input)
        splitter.addWidget(self._result)
        splitter.setSizes([480, 720])
        bl.addWidget(splitter, 1)

        vl.addWidget(body, 1)

    # ── Slots ──
    def _on_dim_change(self):
        self._input.rebuild(self._sidebar.n_vars, self._sidebar.n_cons)
        self._result.show_empty()

    def _reset(self):
        self._input.rebuild(self._sidebar.n_vars, self._sidebar.n_cons)
        self._result.show_empty()

    def _solve(self):
        data = self._input.get_data()
        try:
            prob = LPProblem(
                objective_coeffs=data['c'],
                constraint_matrix=data['A'],
                rhs_values=data['b'],
                constraint_types=data['types'],
                variable_restrictions=data['restrictions'],
                is_maximization=self._sidebar.is_max,
            )
        except Exception as e:
            QMessageBox.critical(self, "Input Error", str(e)); return

        self._solve_btn.setEnabled(False)
        self._solve_btn.setText("Solving…")

        self._worker = SolverThread(prob, self._sidebar.method)
        self._worker.done.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_done(self, result: dict):
        self._solve_btn.setEnabled(True)
        self._solve_btn.setText("▶  Solve")
        self._result.show_result(result)

    def _on_error(self, msg: str):
        self._solve_btn.setEnabled(True)
        self._solve_btn.setText("▶  Solve")
        QMessageBox.critical(self, "Solver Error", msg)


# ══════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()  # ← assign to variable, not anonymous
    window.show()
    window.raise_()
    window.activateWindow()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()