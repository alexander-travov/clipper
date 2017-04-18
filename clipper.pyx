#! -*- coding: utf-8 -*-

from __future__ import division

cimport cython
import numpy as np
cimport numpy as np

# Положение точки относительно направленного отрезка:
# LEFT    - слева от прямой
# RIGHT   - справа от прямой
# BEHIND  - на прямой, позади отрезка
# BETWEEN - на прямой, на отрезке
# BEYOND  - на прямой, за отрезком
LEFT, RIGHT, BEHIND, BETWEEN, BEYOND = range(5)


class Point:
    """
    Класс может использоваться для обозначения точки или вектора на плоскости.
    """
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __repr__(self):
        return 'Point({}, {})'.format(self.x, self.y)

    def scale(self, k):
        return Point(k * self.x, k * self.y)

    def length(self):
        return (self.x * self.x + self.y * self.y)**.5

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def vecdot(self, other):
        return self.x * other.y - self.y * other.x

    def classify(self, start, stop):
        """
        Положение точки относительно прямой, заданной двумя точками:
        слева, справа, позади, между точками, впереди
        """
        a = stop - start
        b = self - start
        orientation = a.vecdot(b)
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
# COLLINEAR     - отрезки сонаправлены, лежат на одной прямой
# PARALLEL      - отрезки лежат на параллельных прямых
# SKEW_CROSS    - отрезки пересекаются
# SKEW_NO_CROSS - прямые на которых лежат отрезки пересекаются, а сами отрезки нет.
COLLINEAR, PARALLEL, SKEW_CROSS, SKEW_NO_CROSS = range(4)


class Edge:
    """
    Направленный отрезок.
    Определяется двумя точками: началом и концом.
    """
    def __init__(self, start, stop):
        self.start = start
        self.stop = stop

    def __repr__(self):
        return 'Edge({}, {})'.format(repr(self.start), repr(self.stop))

    def direction(self):
        """
        Вектор направления отрезка
        """
        return self.stop - self.start

    def normal(self):
        """
        Вектор перпендикулярный к отрезку
        """
        d = self.direction()
        return Point(d.y, -d.x)

    def point(self, t):
        """
        Точка при параметрическом задании прямой, определенной отрезком.
        """
        return self.start + self.direction().scale(t)

    def classify(self, point):
        """
        Положение точки относительно прямой, заданной отрезком:
        слева, справа, позади, между точками, впереди
        """
        return point.classify(self.start, self.stop)

    def intersect(self, other):
        """
        Определяет взаимное расположение двух отрезков и точку пересечения, если они пересекаются.
        """
        n = other.normal()
        denom = self.direction().dot(n)
        if denom:
            t = n.dot(other.start - self.start) / denom
            p = self.point(t)
            if 0 <= t <= 1:
                return SKEW_CROSS, p
            else:
                return SKEW_NO_CROSS, p
        position = other.classify(self.start)
        if position == LEFT or position == RIGHT:
            return PARALLEL, Point()
        return COLLINEAR, Point()

    def aims_at(self, other):
        """
        Определяет нацелен ли отрезок на другой.

        Отрезок считается нацеленным на второй, если прямая, заданная вторым отрезком располагается
        перед первым.
        """
        cross_type, _ = self.intersect(other)
        stop_position = other.classify(self.stop)
        if cross_type == COLLINEAR:
            return stop_position != BEYOND
        if self.direction().vecdot(other.direction()) >= 0:
            return stop_position != RIGHT
        else:
            return stop_position != LEFT


class Polygon:
    """
    Многоугольник
    """
    def __init__(self, points=None):
        if points is None:
            points = []
        self.points = points
        # текущее окно в многоугольнике
        self.current = 0

    def __repr__(self):
        return 'Polygon(' + ', '.join(repr(p) for p in self.points) + ')'

    @property
    def N(self):
        """
        Количество точек в многоугольнике.
        """
        return len(self.points)

    def v(self, n=None):
        """
        Вершина с данным номером
        """
        if n is None:
            n = self.current
        return self.points[n % self.N]

    def e(self, n=None):
        """
        Ребро с данным номером
        """
        if n is None:
            n = self.current
        return Edge(self.v(n), self.v(n+1))

    def add(self, point):
        if not self.N or point != self.v():
            self.points.append(point)
            self.current = self.N - 1

    def advance(self):
        """
        Продвигает текущее окно в многоугольнике
        """
        self.current = (self.current + 1) % self.N

    def contains(self, point):
        """
        Определяет лежит ли точка в многоугольнике.
        """
        for i in range(self.N):
            edge = self.e(i)
            if (edge.classify(point) == LEFT):
                return False
        return True

    def area(self):
        """
        Определяет площадь многоугольника по алгоритму шнурования.
        """
        s = 0
        for i in range(self.N):
            p = self.v(i)
            np = self.v(i+1)
            s += p.x * np.y - p.y * np.x
        return s / 2

    def change_orientation(self):
        return Polygon(list(reversed(self.points)))


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
        p_pos = q.classify(p.stop)
        q_pos = p.classify(q.stop)
        cross_type, intersection_point = p.intersect(q)
        if cross_type == SKEW_CROSS:
            if not main_phase:
                main_phase = True
                R = Polygon()
                R.add(intersection_point)
                start_point = intersection_point
            elif not intersection_point == R.v():
                if intersection_point == start_point:
                    return R
                else:
                    R.add(intersection_point)
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
                R.add(P.v())
        elif q_aims_p:
            Q.advance()
            if flag == OUTSIDE:
                R.add(Q.v())
        else:
            if flag == OUTSIDE or (flag == UNKNOWN and p_pos == LEFT):
                P.advance()
            else:
                Q.advance()

    if P.contains(Q.v()):
        return Q
    elif Q.contains(P.v()):
        return P
    return Polygon()
