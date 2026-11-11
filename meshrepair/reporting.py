"""Russian, escaped HTML report; independent of Qt for regression tests."""
from html import escape
from .reduction import reason_text


def report_html(report):
    lines = ['<style>body{color:#dce8f0;font-family:Segoe UI} h3{color:#64ddc0;margin-bottom:6px} p{margin:5px 0} .warning{color:#efc96b} .bad{color:#ff8e91}</style>']
    def line(text, kind=''):
        lines.append(f'<p class="{kind}">{escape(str(text))}</p>')
    def section(text):
        lines.append('<h3>'+escape(text)+'</h3>')
    def fmt(v):
        return '—' if v is None else (f'{v:.6g}' if isinstance(v, float) else f'{v:,}'.replace(',', ' ') if isinstance(v, int) else str(v))
    before = report.get('before', report.get('info', {}))
    if not before:
        return '<p>Нажмите «Проверить» или «Исправить сетку».</p>'
    section('Исходник')
    for title, key in [('Треугольников', 'triangles'), ('Связных фрагментов после загрузки', 'components'),
                       ('Граней в крупнейшем', 'largest_component_triangles'), ('Мелких частей (<50 граней)', 'small_components'),
                       ('Замкнутых / открытых', None), ('Дыр', 'boundary_loops'), ('Несогласованных рёбер', 'disoriented_edges'),
                       ('Самопересекающихся граней', 'self_intersecting_faces'), ('Вырожденных граней', 'degenerate_faces'),
                       ('Пропущено загрузчиком', 'skipped_faces'), ('Вершин с NaN/inf', 'non_finite_vertices')]:
        value = fmt(before.get(key)) if key else fmt(before.get('closed_components'))+' / '+fmt(before.get('open_components'))
        problem = key in {'boundary_loops','disoriented_edges','self_intersecting_faces','degenerate_faces','skipped_faces','non_finite_vertices'} and before.get(key, 0)
        line(('⚠ ' if problem else '✓ ')+title+': '+value, 'warning' if problem else '')
    orientation = report.get('orientation', {})
    section('Согласование ориентации')
    if orientation:
        line(f"{'✓' if orientation.get('accepted') else '⚠'} Связные части: {orientation.get('components_before')} → {orientation.get('components_after')}; {fmt(orientation.get('elapsed_seconds'))} с.")
        line('Склейка близких вершин: '+ ('включена, '+fmt(orientation['approximate_weld_mm'])+' мм' if orientation.get('approximate_weld_mm') else 'выключена; только точные совпадения.'))
        if orientation.get('reason'):
            line(orientation['reason'], 'warning')
    if orientation.get('conflicting_faces'):
        line(f"⚠ Неориентируемых компонентов: {orientation.get('unoriented_components')}; граней: {orientation['conflicting_faces']}. Они оставлены без согласования.", 'warning')
    if report.get('steps'):
        section('Шаги ремонта')
        names = {'surface_repair': 'Поверхностный ремонт', 'solid_union': 'Булево объединение',
                 'local_cut_and_fill': 'Локальный ремонт CutAndFill',
                 'closed_components_preserved': 'Сохранение замкнутых тел', 'sparse_openvdb_winding': 'Воксельное восстановление'}
        for step in report['steps']:
            line('✓ '+step['name']+': '+fmt(step.get('elapsed_seconds'))+' с')
        line(names.get(report.get('algorithm'), report.get('algorithm')))
        line(report.get('algorithm_reason', ''))
        if report.get('voxel_size_mm'):
            line('Шаг вокселей: '+fmt(report['voxel_size_mm'])+' мм')
    holes = report.get('hole_filling')
    if holes:
        line(f"Дыры до / после заполнения: {holes['holes_before']} / {holes['holes_after']}; отклонено заплаток: {holes['rejected']}; пропущено: {holes['skipped']}.")
        if holes.get('accepted_with_local_fix'):
            line(f"Заплаток принято после локального исправления: {holes['accepted_with_local_fix']}; "
                 f"пересекающихся граней: {holes['intersections_before_local_fix']} → {holes['intersections_after_local_fix']}.")
        if holes.get('rolled_back_local_fix'):
            line(f"Отменено условных заплаток: {holes['rolled_back_local_fix']}; "
                 f"восстановление безопасных заплаток: {fmt(holes['rollback_seconds'])} с.", 'warning')
    sharp = report.get('sharpening')
    if sharp:
        line('Восстановление острых рёбер: '+('принято' if sharp.get('accepted') else 'отклонено' if sharp.get('requested') else 'выключено'), '' if sharp.get('accepted') else 'warning')
        if sharp.get('reason'):
            line(sharp['reason'], 'warning')
        fallback = sharp.get('fallback')
        if fallback:
            line(f"Повтор без Sharpening: оценка {fmt(fallback['estimated_seconds'])} с; "
                 + (f"затрачено {fmt(fallback['elapsed_seconds'])} с." if fallback['attempted']
                    else 'ожидается нажатие «Повторить без Sharpening».'))
    manual = report.get('manual_sharpening_retry')
    if manual:
        line('Повтор без Sharpening по кнопке: '+fmt(manual['elapsed_seconds'])+' с.')
        line(manual['previous_attempt'].get('reason', ''), 'warning')
    local = report.get('local_repair', {})
    if local.get('attempted'):
        section('Локальный ремонт')
        line('CutAndFill: '+('принят' if local.get('accepted') else 'отклонён'))
        line(f"Самопересекающиеся грани: {local['before']['self_intersecting_faces']} → {local.get('after', {}).get('self_intersecting_faces', '—')}")
        for shell in local.get('shells', []):
            xyz = ', '.join(fmt(v) for v in shell['center_mm'])
            line(f"Оболочка {shell['id']}: {shell['triangles']} граней; центр bbox ({xyz}) мм.")
        deviation = local.get('source_deviation', {})
        if deviation.get('top10'):
            line(f"После CutAndFill: максимум {fmt(deviation['max_deviation_mm'])} мм; "
                 f"вершин > {fmt(deviation['warning_threshold_mm'])} мм: {deviation['exceeding_vertices']}.")
            line('Наихудшая точка исходника: '+', '.join(fmt(v) for v in deviation['top10'][0]['coordinates_mm'])+' мм.')
    for shell in report.get('removed_shells', []):
        line(f"Удалена мелкая оболочка: {shell['triangles']} граней, центр {shell['center_mm']} мм.", 'warning')
    after = report.get('after')
    if after:
        section('Результат')
        line(f"✓ Треугольников {fmt(after['triangles'])}; оболочек {after['components']}; замкнутость: {'да' if after['closed'] else 'нет'}.")
        line('Объём: '+fmt(after.get('signed_volume_mm3'))+' мм³')
        deviation = report.get('source_deviation', {})
        line('Макс. отклонение исходник → результат: '+fmt(deviation.get('max_deviation_mm'))+' мм (измерено по вершинам исходника, не доказано для всей поверхности).')
        if deviation.get('top10'):
            line(f"Вершин с отклонением > двух шагов: {deviation.get('exceeding_vertices', 0)}. Кнопка «Отклонения» показывает карту в 3D.", 'warning' if deviation.get('exceeding_vertices') else '')
            line('Наибольшие отклонения — координаты исходника в мм:')
            for spot in deviation['top10']:
                xyz = ', '.join(fmt(v) for v in spot['coordinates_mm'])
                line(f"({xyz}): {fmt(spot['deviation_mm'])} мм")
    reduction = report.get('simplification')
    section('Упрощение')
    if reduction:
        input_count = reduction.get('input_triangles')
        if input_count is not None:
            line(f"Исходник → после ремонта → после упрощения: {fmt(input_count)} → {fmt(reduction['before'])} → {fmt(reduction['after'])} треугольников.")
        accepted = reduction.get('accepted')
        percent = 100*(1-reduction['after']/max(1, reduction['before']))
        line(f"{'✓ Принято' if accepted else '✗ Отклонено'}: {reduction['before']:,} → {reduction['after']:,} (−{percent:.1f}%)", '' if accepted else 'bad')
        verification = reduction.get('verification', {})
        mode = verification.get('mode', reduction.get('mode'))
        if mode:
            line('Проверка: '+ ('доказано численным ограничением всей поверхности' if mode == 'proven' else 'измерено по вершинам в обе стороны; не доказано'), '' if mode == 'proven' else 'warning')
            line('Отклонение от исправленной сетки: '+fmt(verification.get('max_deviation_mm'))+' мм')
        for attempt in reduction.get('attempts', []):
            text = f"Попытка {attempt['attempt']}: погрешность алгоритма {fmt(attempt['max_error_mm'])} мм; {fmt(attempt.get('elapsed_seconds'))} с"
            if attempt.get('reason'):
                text += '. '+reason_text(attempt['reason'])
            line(text)
        if reduction.get('stages'):
            line('Треугольники после проходов: '+ ' → '.join(
                fmt(stage.get('after_with_protected', stage['after'])) for stage in reduction['stages']))
        if not accepted:
            line(reduction.get('reason_ru', reason_text(reduction.get('reason','native_error'))), 'bad')
    else:
        line('Не запрошено.' if after else 'Выполняется при ремонте, если включено в настройках.')
    distribution = report.get('triangle_distribution')
    if distribution:
        section('Где остались треугольники')
        line(f"Большие плоскости: {fmt(distribution['large_planes_triangles'])} "
             f"({distribution['large_planes_fraction']:.1%}); остальная поверхность: "
             f"{fmt(distribution['other_triangles'])} ({distribution['other_fraction']:.1%}).")
        line('Оценка по кластерам нормалей ±8° площадью > 3 мм²; рёбра, скругления и шум могут попадать в остальную поверхность.')
        for plane in distribution['planes']:
            normal = ', '.join(f'{v:.2f}' for v in plane['normal'])
            line(f"Плоскость {plane['id']} (нормаль {normal}): {plane['triangles']} треугольников, "
                 f"{plane['fraction']:.1%}; остаток исходника p50 / p90: "
                 f"{fmt(plane['source_residual_p50_mm'])} / {fmt(plane['source_residual_p90_mm'])} мм.")
            if plane['rougher_than_tolerance']:
                line('Грань шершавая сильнее допуска, полигоны на ней сохраняются; увеличьте допуск.', 'warning')
        line('Эвристическая подсказка: при допуске ×2,5 возможно примерно −25% треугольников. '
             'Без перерасчёта; результат зависит от формы и шума.')
    section('Предупреждения')
    for warning in report.get('warnings', []):
        line('⚠ '+warning, 'warning')
    if not report.get('warnings'):
        line('✓ Нет зарегистрированных предупреждений. Это не сертификат точности ремонта.')
    section('Ускорение')
    cap = report.get('acceleration', {})
    line(f"CPU: {cap.get('cpu_threads','авто')} доступных потоков; топология, булевы операции и decimation — нативный CPU.")
    line('Ориентация: NumPy + SciPy (нативный граф, без Python BFS).')
    line('Восстановление: '+{'cpu_native_parallel':'нативный CPU','cuda_winding_with_cpu_openvdb':'CUDA winding + CPU OpenVDB'}.get(report.get('compute_device'),'не выполнялось'))
    line('Проекции: '+', '.join(cap.get('projection_backends', [])) if cap.get('projection_backends') else 'Проекции: нативный CPU для измерений; CUDA при доступности для строгой проверки.')
    return '\n'.join(lines)
