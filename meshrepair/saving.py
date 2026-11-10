"""Russian save dialog. File writes live in storage.save_artifacts, not widgets."""
from pathlib import Path
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
                              QPushButton, QCheckBox, QDialogButtonBox, QFileDialog)


class SaveDialog(QDialog):
    def __init__(self, default_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Сохранить результат')
        self.resize(620, 270)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Путь и общее имя файлов результата:'))
        row = QHBoxLayout()
        self.path = QLineEdit(str(default_path))
        row.addWidget(self.path)
        browse = QPushButton('Обзор…')
        browse.clicked.connect(self.browse)
        row.addWidget(browse)
        layout.addLayout(row)
        self.choices = {}
        for key, title in [('stl', 'STL результата'), ('json', 'JSON-отчёт'),
                           ('deviation', 'Карта отклонения (.deviation.npz)'), ('log', 'Журнал (.log.txt)')]:
            checkbox = QCheckBox(title)
            checkbox.setChecked(key == 'stl')
            self.choices[key] = checkbox
            layout.addWidget(checkbox)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).setText('Сохранить')
        buttons.button(QDialogButtonBox.Cancel).setText('Отмена')
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def browse(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Сохранить результат', self.path.text(),
            'STL (*.stl)', options=QFileDialog.DontConfirmOverwrite)
        if filename:
            path = Path(filename)
            self.path.setText(str(path if path.suffix else path.with_suffix('.stl')))

    def selection(self):
        return {key: checkbox.isChecked() for key, checkbox in self.choices.items()}
