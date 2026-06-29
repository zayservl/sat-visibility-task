## Структура репозитория

```
sat_visibility.py          # библиотека математических функций
test_sat_visibility.py     # 21 модульный тест (pytest)
solution_executed.ipynb    # ноутбук с основным расчётом
```

## Использование

```python
from sat_visibility import compute_elevation, is_visible
import numpy as np

position_eci = np.array([4435144.0, -2137297.0, 4670064.0])
jd_tt = 2451545.0 + 8084.185608609847

elevation = compute_elevation(position_eci, jd_tt, 45.920266, -63.342286, delta_t=69.29)
visible = is_visible(elevation, min_elevation=15.0)
```

## Тесты

```bash
pytest test_sat_visibility.py
```

## Зависимости

- numpy
- matplotlib (для визуализации в ноутбуке)