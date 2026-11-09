"""Qt layout and lightweight model for shell diagnostics."""
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QLocale
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
    QFrame, QDoubleSpinBox, QCheckBox, QProgressBar, QPlainTextEdit, QSplitter,
    QTabWidget, QTableView, QHeaderView, QScrollArea)
from .viewer import MeshView


def label(text, name=None, wrap=False):
    widget = QLabel(text)
    widget.setTextFormat(Qt.PlainText)
    widget.setWordWrap(wrap)
    if name:
        widget.setObjectName(name)
    return widget


def button(text, callback, name=None):
    widget = QPushButton(text)
    if name:
        widget.setObjectName(name)
    widget.clicked.connect(callback)
    return widget


def card(name='card', margins=16):
    frame = QFrame()
    frame.setObjectName(name)
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(margins, margins, margins, margins)
    layout.setSpacing(10)
    return frame, layout


class ShellModel(QAbstractTableModel):
    headers = ('Оболочка', 'Треугольники', 'Замкнута', 'Открытые рёбра', 'Размер bbox, мм', 'Тип')

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.headers[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        row = self.rows[index.row()]
        values = (row['id'], f"{row['triangles']:,}".replace(',', ' '),
                  'Да' if row['closed'] else 'Нет', row['boundary_edges'],
                  ' × '.join(f'{x:.3f}' for x in row.get('bbox_mm', [])), row.get('kind', ''))
        if role == Qt.DisplayRole:
            return str(values[index.column()])
        if role == Qt.ForegroundRole and index.column() == 2:
            return QColor('#78dfb9' if row['closed'] else '#f3b0a6')
        return None

    def set_rows(self, rows):
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()


class DecimalBox(QDoubleSpinBox):
    """Accept both decimal separators, independent of Windows regional settings."""
    def normalized(self, text):
        point = self.locale().decimalPoint()
        return text.replace('.', point).replace(',', point)

    def validate(self, text, position):
        return super().validate(self.normalized(text), position)

    def valueFromText(self, text):
        return super().valueFromText(self.normalized(text))


class LayoutMixin:
    def setup_ui(self, preview_enabled):
        root = QWidget()
        root.setObjectName('root')
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(22, 18, 22, 18)
        outer.setSpacing(14)
        header = QHBoxLayout()
        monogram = label('MR', 'monogram')
        monogram.setFixedSize(48, 48)
        monogram.setAlignment(Qt.AlignCenter)
        header.addWidget(monogram)
        brand = QVBoxLayout()
        brand.setSpacing(1)
        brand.addWidget(label('MeshRepair', 'brand'))
        brand.addWidget(label('Проверка, восстановление и оптимизация сеток', 'subtitle'))
        header.addLayout(brand)
        header.addStretch()
        self.hardware = label('CPU / CUDA · авто', 'badge')
        header.addWidget(self.hardware)
        header.addWidget(label('0.7', 'muted'))
        outer.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(16)
        left = QVBoxLayout()
        left.setSpacing(12)
        sidebar = QWidget()
        sidebar.setLayout(left)
        sidebar.setFixedWidth(300)
        left.setContentsMargins(0, 0, 0, 0)
        drop, layout = card('dropCard')
        layout.addWidget(label('01  ИСХОДНЫЙ ФАЙЛ', 'section'))
        self.filename = label('Перетащите сетку\nв окно приложения', wrap=True)
        self.filename.setMinimumHeight(46)
        self.filename.setMaximumHeight(56)
        self.filename.setFont(QFont('Segoe UI', 14, QFont.DemiBold))
        layout.addWidget(self.filename)
        layout.addWidget(label('STL · OBJ · PLY  /  миллиметры', 'muted'))
        self.browse = button('Выбрать файл…', self.pick_source)
        layout.addWidget(self.browse)
        left.addWidget(drop)
        settings, layout = card()
        layout.addWidget(label('02  ПАРАМЕТРЫ РЕМОНТА', 'section'))
        layout.addWidget(label('Качество / допуск'))
        self.quality = DecimalBox()
        self.quality.setLocale(QLocale('ru_RU'))
        self.quality.setRange(.001, .5)
        self.quality.setDecimals(3)
        self.quality.setSingleStep(.001)
        self.quality.setSuffix(' мм')
        self.quality.setValue(.02)
        layout.addWidget(self.quality)
        layout.addWidget(label('Меньше — подробнее.\nБольше — быстрее и грубее.', 'muted', True))
        layout.addSpacing(4)
        self.planar = QCheckBox('Схлопывать копланарные\nполигоны')
        layout.addWidget(self.planar)
        angle_row = QHBoxLayout()
        angle_row.addWidget(label('Угол копланарности', 'muted'))
        self.angle = DecimalBox()
        self.angle.setLocale(QLocale('ru_RU'))
        self.angle.setRange(0, 45)
        self.angle.setDecimals(1)
        self.angle.setSingleStep(.1)
        self.angle.setSuffix('°')
        self.angle.setValue(1)
        self.angle.setFixedWidth(88)
        self.angle.setEnabled(False)
        self.planar.toggled.connect(lambda _: self.update_reduction_controls())
        angle_row.addWidget(self.angle)
        layout.addLayout(angle_row)
        self.curved = QCheckBox('Упрощать всю поверхность\nв пределах допуска')
        self.curved.setToolTip('Включает плоскости и криволинейные участки одним проходом. Угол копланарности применяется только при выключенном общем упрощении.')
        self.curved.toggled.connect(lambda _: self.update_reduction_controls())
        self.angle.setToolTip('Порог резких переходов для отдельного схлопывания плоскостей. При общем упрощении форму ограничивает допуск в миллиметрах.')
        layout.addWidget(self.curved)
        self.strict = QCheckBox('Строго доказывать допуск')
        self.strict.setToolTip('Выкл.: допускается измерение по вершинам. Вкл.: требуется численное ограничение всей поверхности.')
        layout.addWidget(self.strict)
        self.weld_close = QCheckBox('Склеивать близкие вершины')
        self.weld_close.setToolTip('Допуск склейки = 0,1 × качество. Может изменить мелкие детали. По умолчанию выключено.')
        layout.addWidget(self.weld_close)
        layout.addWidget(label('Допуск упрощения отсчитывается\nот уже исправленной поверхности.', 'muted', True))
        left.addWidget(settings)
        footer = QVBoxLayout()
        footer.setContentsMargins(0, 0, 16, 0)
        footer.setSpacing(10)
        self.repair = button('Исправить сетку', lambda: self.start('repair'), 'primary')
        footer.addWidget(self.repair)
        actions = QHBoxLayout()
        self.check_button = button('Check / Проверить', lambda: self.start('check'))
        self.stop_button = button('Остановить', self.cancel, 'danger')
        actions.addWidget(self.check_button)
        actions.addWidget(self.stop_button)
        footer.addLayout(actions)
        footer.addWidget(label('Результат — новый STL рядом\nс исходником. Исходник сохраняется.', 'muted', True))
        left.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(316)
        scroll.setWidget(sidebar)
        scroll.setStyleSheet('QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }')
        side_column = QVBoxLayout()
        side_column.addWidget(scroll, 1)
        side_column.addLayout(footer)
        body.addLayout(side_column)

        right = QVBoxLayout()
        right.setSpacing(12)
        stats = QHBoxLayout()
        self.stat_labels = {}
        for key, title in [('triangles', 'ТРЕУГОЛЬНИКИ'), ('components', 'СВЯЗНЫЕ ЧАСТИ'), ('closed', 'ЗАМКНУТОСТЬ')]:
            frame, layout = card(margins=12)
            value = label('—', 'statValue')
            layout.addWidget(value)
            layout.addWidget(label(title, 'statLabel'))
            if key == 'components':
                frame.setToolTip('Связные части топологии после загрузки. Перевёрнутые грани могут отделяться загрузчиком; это не число физических тел. Check покажет эффект согласования.')
            self.stat_labels[key] = value
            stats.addWidget(frame)
        right.addLayout(stats)
        split = QSplitter(Qt.Vertical)
        viewport, layout = card(margins=10)
        toolbar = QHBoxLayout()
        self.source_button = button('Исходник', lambda: self.show_source(False))
        self.result_button = button('Результат', lambda: self.show_source(True))
        for b in (self.source_button, self.result_button):
            b.setCheckable(True)
            toolbar.addWidget(b)
        self.source_button.setChecked(True)
        toolbar.addStretch()
        self.surface_button = button('Поверхность', self.update_view)
        self.edge_button = button('Рёбра', self.update_view)
        for b in (self.surface_button, self.edge_button):
            b.setCheckable(True)
            toolbar.addWidget(b)
        self.surface_button.setChecked(True)
        layout.addLayout(toolbar)
        self.view = MeshView(enabled=preview_enabled)
        self.view.setMinimumHeight(180)
        self.view.renderer_info.connect(self.show_renderer)
        layout.addWidget(self.view, 1)
        foot = QHBoxLayout()
        self.render_label = label('ЛКМ: вращение · СКМ: сдвиг · колесо: масштаб', 'muted')
        self.render_label.setStyleSheet('font-size: 11px;')
        foot.addWidget(self.render_label, 1)
        for title, view in [('Вписать', None), ('Изо', 'iso'), ('Сверху', 'top'), ('Спереди', 'front')]:
            foot.addWidget(button(title, lambda checked=False, v=view: self.view.fit(v), 'flat'))
        layout.addLayout(foot)
        split.addWidget(viewport)
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        log_panel = QWidget()
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(0, 8, 0, 0)
        log_tools = QHBoxLayout()
        self.check_state = label('Пересечения ещё не проверены', 'muted')
        log_tools.addWidget(self.check_state, 1)
        self.copy_button = button('Копировать', self.copy_logs, 'flat')
        self.clear_button = button('Очистить', self.clear_logs, 'flat')
        log_tools.addWidget(self.copy_button)
        log_tools.addWidget(self.clear_button)
        log_layout.addLayout(log_tools)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setAcceptDrops(False)
        self.log.setPlaceholderText('Здесь появятся результаты проверки и ход обработки.\nТекст можно выделить и скопировать: Ctrl+C.')
        log_layout.addWidget(self.log)
        self.tabs.addTab(log_panel, 'Журнал')
        details = QWidget()
        details_layout = QVBoxLayout(details)
        self.detail_text = label('Нажмите «Check / Проверить» для подробной диагностики.', 'muted', True)
        details_layout.addWidget(self.detail_text)
        self.shell_model = ShellModel()
        self.shell_table = QTableView()
        self.shell_table.setModel(self.shell_model)
        self.shell_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.shell_table.verticalHeader().hide()
        self.shell_table.setAlternatingRowColors(True)
        details_layout.addWidget(self.shell_table)
        self.tabs.addTab(details, 'Сведения и оболочки')
        from PySide6.QtWidgets import QTextBrowser
        self.report_view = QTextBrowser()
        self.report_view.setOpenExternalLinks(False)
        self.report_view.setStyleSheet('QTextBrowser { background: #101b29; color: #dce8f0; border: 1px solid #293a50; border-radius: 8px; padding: 10px; }')
        self.report_view.setHtml('<p>Нажмите Check или Исправить для отчёта.</p>')
        self.tabs.addTab(self.report_view, 'Отчёт')
        bottom_layout.addWidget(self.tabs)
        split.addWidget(bottom)
        split.setSizes([450, 220])
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        right.addWidget(split, 1)
        body.addLayout(right, 1)
        outer.addLayout(body, 1)
        status_row = QHBoxLayout()
        self.status = label('Выберите или перетащите файл.', 'muted')
        self.elapsed = label('', 'muted')
        status_row.addWidget(self.status, 1)
        status_row.addWidget(self.elapsed)
        outer.addLayout(status_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(7)
        outer.addWidget(self.progress)
