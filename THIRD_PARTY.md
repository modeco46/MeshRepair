# Сторонние компоненты

- **MeshLib / meshlib-core 3.1.4.297** — MeshInspector, нативная обработка сеток и CUDA bindings. [Репозиторий и условия](https://github.com/MeshInspector/MeshLib). Source-available; не считать свободной библиотекой для любого коммерческого использования. Коммерческие права оформляются у производителя.
- **Python** — среда исполнения. [Python license](https://docs.python.org/3/license.html).
- **PySide6-Essentials / Qt 6.8.3 и Shiboken6** — интерфейс. [Условия Qt for Python](https://doc.qt.io/qtforpython-6/licenses.html). Применимы условия LGPL/GPL/коммерческой лицензии соответствующих компонентов.
- **VTK 9.5.2** — геометрия предпросмотра и OpenGL. [Лицензия VTK](https://vtk.org/about/#license).
- **NumPy** — зависимость MeshLib. [License](https://numpy.org/doc/stable/license.html).
- **PyInstaller** — упаковка Windows-приложения. [License](https://pyinstaller.org/en/stable/license.html).

При распространении сборок необходимо сохранять условия и уведомления поставляемых зависимостей. Сборка использует `--collect-all meshlib`, включая доступные данные пакета; это не заменяет проверку условий распространения владельцем приложения. Получение бинарников через pip не предоставляет дополнительных коммерческих прав.

- SciPy 1.16.3 — BSD-3-Clause; нативные разреженные графы ориентации: https://scipy.org/ и https://github.com/scipy/scipy/blob/v1.16.3/LICENSE.txt
