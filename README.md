Clipper
=======

Simple python module for finding convex polygons intersection area.

Based on algorithm from `Ласло. Вычислительная геометрия и компьютерная графика на c++`.

``` python
    import numpy as np
    from clipper import *
    P = Polygon(np.array([-2., -2., 2., 2.]), np.array([-2., 2., 2., -2.]))
    Q = Polygon(np.array([0., -3., 0., 3.]), np.array([-3., 0., 3., 0.]))
    intersection_area(P, Q)
    # 14
```
