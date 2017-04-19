Clipper
=======

Simple python module for finding convex polygons intersection.

Based on algorithm from `Ласло. Вычислительная геометрия и компьютерная графика на c++`.

``` python
    import numpy as np
    from clipper import *
    P = Polygon([Point(2,2), Point(2,-2), Point(-2,-2), Point(-2,2)])
    Q = Polygon([Point(0,3), Point(3, 0), Point(0,-3), Point(-3,0)])
    intersection(P, Q)
    # Polygon(Point(2.0, 1.0), Point(2.0, -1.0), Point(1.0, -2.0), Point(-1.0, -2.0),
    #         Point(-2.0, -1.0), Point(-2.0, 1.0), Point(-1.0, 2.0), Point(1.0, 2.0))
```
