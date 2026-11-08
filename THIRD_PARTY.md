# Сторонние компоненты

- **MeshLib / meshlib-core 3.1.4.297** — MeshInspector, нативная обработка сеток и CUDA bindings. [Репозиторий и условия](https://github.com/MeshInspector/MeshLib). Source-available; не считать свободной библиотекой для любого коммерческого использования. Коммерческие права оформляются у производителя.
- **Python / Tcl/Tk** — среда исполнения и интерфейс. [Python license](https://docs.python.org/3/license.html).
- **NumPy** — зависимость MeshLib. [License](https://numpy.org/doc/stable/license.html).
- **PyInstaller** — упаковка Windows-приложения. [License](https://pyinstaller.org/en/stable/license.html).

При распространении сборок необходимо сохранять условия и уведомления поставляемых зависимостей. Сборка использует `--collect-all meshlib`, включая доступные данные пакета; это не заменяет проверку условий распространения владельцем приложения. Получение бинарников через pip не предоставляет дополнительных коммерческих прав.
