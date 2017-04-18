#! -*- coding: utf-8 -*-

from __future__ import division

cimport cython
from cpython.object cimport Py_EQ, Py_NE
import numpy as np
cimport numpy as np


# Положение точки относительно направленного отрезка:
cpdef enum PointOrientation:
    LEFT = 1     # слева от прямой
    RIGHT = 2    # справа от прямой
    BEHIND = 3   # на прямой, позади отрезка
    BETWEEN = 4  # на прямой, на отрезке
    BEYOND = 5   # на прямой, за отрезком


cdef class Point:
    """
    Класс может использоваться для обозначения точки или вектора на плоскости.
    """
    cdef readonly float x, y

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    cpdef Point add(Point self, Point other):
        return Point(self.x + other.x, self.y + other.y)

    def __add__(self, other):
        return self.add(other)

    cpdef Point sub(self, Point other):
        return Point(self.x - other.x, self.y - other.y)

    def __sub__(self, other):
        return self.sub(other)

    def __richcmp__(self, other, operation):
        if operation == Py_EQ:
            return self.x == other.x and self.y == other.y
        elif operation == Py_NE:
            return self.x != other.x or self.y != other.y
        else:
            return False

    def __repr__(self):
        return 'Point({}, {})'.format(self.x, self.y)

    cpdef Point scale(self, double k):
        return Point(k * self.x, k * self.y)

    cpdef double length(self):
        return (self.x * self.x + self.y * self.y)**.5

    cpdef double dot(self, Point other):
        return self.x * other.x + self.y * other.y

    cpdef double vecdot(self, Point other):
        return self.x * other.y - self.y * other.x

    cpdef PointOrientation classify(self, Point start, Point stop):
        """
        Положение точки относительно прямой, заданной двумя точками:
        слева, справа, позади, между точками, впереди
        """
        cdef Point a = stop.sub(start), b = self.sub(start)
        cdef double orientation = a.vecdot(b)
        if orientation > 0:
            return LEFT
        if orientation < 0:
            return RIGHT
        if a.x * b.x < 0 or a.y * b.y < 0:
            return BEHIND
        if a.length() < b.length():
            return BEYOND
        return BETWEEN


# Взаимное расположение двух отрезков:
cpdef enum LineOrientation:
    COLLINEAR = 1 # отрезки сонаправлены, лежат на одной прямой
    PARALLEL = 2 # отрезки лежат на параллельных прямых
    SKEW_CROSS = 3 # отрезки пересекаются
    SKEW_NO_CROSS = 4 # прямые на которых лежат отрезки пересекаются, а сами отрезки нет.


cdef class Edge:
    """
    Направленный отрезок.
    Определяется двумя точками: началом и концом.
    """
    cdef readonly Point start, stop

    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __repr__(self):
        return 'Edge({}, {})'.format(repr(self.start), repr(self.stop))

    def __richcmp__(self, other, operation):
        if operation == Py_EQ:
            return self.start == other.start and self.stop == other.stop
        if operation == Py_NE:
            return self.start != other.start or self.stop != other.stop
        return False

    cpdef Point direction(self):
        """
        Вектор направления отрезка
        """
        return self.stop.sub(self.start)

    cpdef Point normal(self):
        """
        Вектор перпендикулярный к отрезку
        """
        cdef Point d = self.direction()
        return Point(d.y, -d.x)

    cpdef Point point(self, double t):
        """
        Точка при параметрическом задании прямой, определенной отрезком.
        """
        return self.start.add(self.direction().scale(t))

    cpdef PointOrientation point_classify(self, Point point):
        """
        Положение точки относительно прямой, заданной отрезком:
        слева, справа, позади, между точками, впереди
        """
        return point.classify(self.start, self.stop)

    cpdef LineOrientation edge_classify(self, Edge other):
        """
        Определяет взаимное расположение двух отрезков и точку пересечения, если они пересекаются.
        """
        cdef Point normal = self.normal(), other_normal = other.normal()
        cdef double denom1 = self.direction().dot(other_normal), t, u
        if denom1:
            # Определяем параметры точки пересечения на отрезках
            t = other_normal.dot(other.start.sub(self.start)) / denom1
            u = normal.dot(self.start.sub(other.start)) / other.direction().dot(normal)
            if t < 0 or t >= 1 or u < 0 or u >= 1:
                return SKEW_NO_CROSS
            else:
                return SKEW_CROSS
        cdef PointOrientation orientation = other.point_classify(self.start)
        if orientation == LEFT or orientation == RIGHT:
            return PARALLEL
        return COLLINEAR

    cpdef Point intersection_point(self, Edge other):
        cdef Point n = other.normal()
        cdef double t = n.dot(other.start.sub(self.start)) / self.direction().dot(n)
        return self.point(t)

    cpdef int aims_at(self, Edge other):
        """
        Определяет нацелен ли отрезок на другой.

        Отрезок считается нацеленным на второй, если прямая, заданная вторым отрезком располагается
        перед первым.
        """
        cdef LineOrientation cross_type = self.edge_classify(other)
        cdef PointOrientation stop_position = other.point_classify(self.stop)
        if cross_type == COLLINEAR:
            return stop_position != BEYOND
        if self.direction().vecdot(other.direction()) >= 0:
            return stop_position != RIGHT
        else:
            return stop_position != LEFT


cdef class Polygon:
    """
    Многоугольник
    """

    cdef readonly double[:] x, y
    cdef readonly int N, current

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.N = x.shape[0]
        # текущее окно в многоугольнике
        self.current = 0

    def __repr__(self):
        return 'Polygon(' + ', '.join(repr(self.v(i)) for i in range(self.N)) + ')'

    cpdef Point v(self, int n=-1):
        """
        Вершина с данным номером
        """
        if n == -1:
            n = self.current
        return Point(self.x[n % self.N], self.y[n % self.N])

    cpdef Edge e(self, n=-1):
        """
        Ребро с данным номером
        """
        if n == -1:
            n = self.current
        return Edge(self.v(n), self.v(n+1))

    cpdef advance(self):
        """
        Продвигает текущее окно в многоугольнике
        """
        self.current = (self.current + 1) % self.N

    cpdef int contains(self, Point point):
        """
        Определяет лежит ли точка в многоугольнике.
        """
        cdef int i = 0
        cdef Edge edge
        for i in range(self.N):
            edge = self.e(i)
            if (edge.point_classify(point) == LEFT):
                return False
        return True

    cpdef area(self):
        """
        Определяет площадь многоугольника по алгоритму шнурования.
        """
        s = 0
        for i in range(self.N):
            p = self.v(i)
            np = self.v(i+1)
            s += triangle_area(p, np)
        return abs(s)


cpdef double triangle_area(Point start, Point stop):
    return (start.x * stop.y - start.y * stop.x)/2


UNKNOWN, INSIDE, OUTSIDE = range(3)
def intersection(P, Q):
    """
    Алгоритм пересечения двух выпуклых многоугольников P, Q.
    Ласло. Вычислительная геометрия и компьютерная графика на c++.
    Сложность по времени линейная по числу вершин: O(|P| + |Q|)

    Полигоны ориентированы против часовой стрелки.
    """
    R = None
    flag = UNKNOWN
    main_phase = False
    max_iterations = 2 * (P.N + Q.N)
    start_point = None

    i = 0
    while i < max_iterations or main_phase:
        i += 1
        p = P.e()
        q = Q.e()
        p_pos = q.point_classify(p.stop)
        q_pos = p.point_classify(q.stop)
        cross_type = p.edge_classify(q)
        if cross_type == SKEW_CROSS:
            intersection_point = p.intersection_point(q)
            if not main_phase:
                main_phase = True
                R = []
                R.append(intersection_point)
                start_point = intersection_point
            elif not intersection_point == R[-1]:
                if intersection_point == start_point:
                    xs = np.array([p.x for p in R], dtype=np.float_)
                    ys = np.array([p.y for p in R], dtype=np.float_)
                    return Polygon(xs, ys)
                else:
                    R.append(intersection_point)
            if p_pos == RIGHT:
                flag = INSIDE
            elif q_pos == RIGHT:
                flag = OUTSIDE
            else:
                flag = UNKNOWN
        elif cross_type == COLLINEAR and p_pos != BEHIND and q_pos != BEHIND:
            flag = UNKNOWN

        p_aims_q = p.aims_at(q)
        q_aims_p = q.aims_at(p)

        if p_aims_q and q_aims_p:
            if flag == OUTSIDE or (flag == UNKNOWN and p_pos == LEFT):
                P.advance()
            else:
                Q.advance()
        elif p_aims_q:
            P.advance()
            if flag == INSIDE:
                R.append(P.v())
        elif q_aims_p:
            Q.advance()
            if flag == OUTSIDE:
                R.append(Q.v())
        else:
            if flag == OUTSIDE or (flag == UNKNOWN and p_pos == LEFT):
                P.advance()
            else:
                Q.advance()

    if P.contains(Q.v()):
        return Q
    elif Q.contains(P.v()):
        return P
    return Polygon(np.array([0.]), np.array([0.]))
