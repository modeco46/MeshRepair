"""Restrained dark desktop palette with native Qt interaction and focus states."""
from pathlib import Path
STYLE = '''
QMainWindow, QWidget#root { background: #0c121c; color: #e8eef7; }
QWidget { color: #e8eef7; font-family: "Segoe UI", "DejaVu Sans"; font-size: 13px; }
QLabel { background: transparent; }
QLabel#brand { font-size: 26px; font-weight: 700; letter-spacing: -1px; }
QLabel#monogram { color: #0c121c; background: #62dfbc; border-radius: 12px; font-size: 20px; font-weight: 800; }
QLabel#subtitle, QLabel#muted { color: #93a4bb; }
QLabel#section { color: #93a4bb; font-size: 11px; font-weight: 700; letter-spacing: 1px; }
QLabel#statValue { font-size: 23px; font-weight: 650; }
QLabel#statLabel { font-size: 11px; color: #93a4bb; }
QLabel#badge { color: #a6cbbf; background: #172c2a; border: 1px solid #2b4841; border-radius: 12px; padding: 5px 10px; }
QLabel#emptyView { color: #74889f; font-size: 17px; background: #111d2b; border-radius: 10px; }
QFrame#card { background: #141e2c; border: 1px solid #253247; border-radius: 12px; }
QFrame#dropCard { background: #15232f; border: 1px dashed #416473; border-radius: 12px; }
QPushButton { background: #1c2a3d; border: 1px solid #304057; border-radius: 7px; padding: 8px 13px; font-weight: 600; }
QPushButton:hover { background: #293b52; border-color: #4c657e; }
QPushButton:pressed { background: #324b62; }
QPushButton:checked { background: #1c4941; border-color: #58cfac; color: #a4f1d7; }
QPushButton:disabled { color: #607085; background: #17212e; border-color: #243044; }
QPushButton#primary { background: #62dfbc; color: #0d2620; border: none; padding: 13px; font-size: 15px; }
QPushButton#primary:hover { background: #8cebcf; }
QPushButton#primary:disabled { background: #274c43; color: #658d80; }
QPushButton#danger { color: #f5b8b8; border-color: #604047; }
QPushButton#danger:disabled { color: #607085; border-color: #243044; }
QPushButton#flat { background: transparent; border: none; color: #a7b9cf; padding: 5px 8px; }
QPushButton#flat:hover { background: #24364b; }
QPushButton:focus, QDoubleSpinBox:focus { border: 1px solid #62dfbc; }
QDoubleSpinBox { background: #0f1826; border: 1px solid #35465e; border-radius: 7px; padding: 9px; font-size: 15px; selection-background-color: #326e60; }
QDoubleSpinBox:disabled { color: #607085; border-color: #263449; }
QCheckBox { spacing: 8px; background: transparent; padding: 4px 0; }
QCheckBox::indicator { width: 17px; height: 17px; }
QCheckBox::indicator:unchecked { background: #0f1826; border: 1px solid #4c6179; border-radius: 4px; }
QCheckBox::indicator:checked { background: #62dfbc; border: 1px solid #62dfbc; border-radius: 4px; image: none; }
QCheckBox:disabled { color: #607085; }
QPlainTextEdit, QTableView { background: #0d1622; border: 1px solid #253247; border-radius: 7px; selection-background-color: #27574e; gridline-color: #243246; }
QPlainTextEdit { font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono"; font-size: 12px; padding: 8px; }
QTableView { alternate-background-color: #131f2f; }
QHeaderView::section { background: #1b2a3d; color: #9eafc4; border: none; padding: 7px; }
QTabWidget::pane { border: none; }
QTabBar::tab { background: transparent; color: #8fa0b6; padding: 9px 15px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #d7f6ec; border-bottom: 2px solid #62dfbc; }
QProgressBar { background: #1c2b3e; border: none; border-radius: 4px; height: 8px; }
QProgressBar::chunk { background: #62dfbc; border-radius: 4px; }
QSplitter::handle { background: #0c121c; height: 8px; width: 8px; }
QScrollBar:vertical { background: #101a28; width: 10px; }
QScrollBar::handle:vertical { background: #35485f; border-radius: 4px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QToolTip { color: #e8eef7; background: #23354b; border: 1px solid #46607c; padding: 6px; }
'''

_assets = Path(__file__).with_name('assets').as_posix()
STYLE += '''
QCheckBox::indicator:checked { image: url("%s/check.svg"); }
QDoubleSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 23px; border-left: 1px solid #35465e; background: #1b2c40; border-top-right-radius: 6px; }
QDoubleSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 23px; border-left: 1px solid #35465e; background: #1b2c40; border-bottom-right-radius: 6px; }
QDoubleSpinBox::up-arrow { image: url("%s/up.svg"); width: 10px; height: 7px; }
QDoubleSpinBox::down-arrow { image: url("%s/down.svg"); width: 10px; height: 7px; }
''' % (_assets, _assets, _assets)
