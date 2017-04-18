from clipper import *


def test_points_addition():
    p1 = Point(1, 1)
    p2 = Point(2, 3)
    p = p1 + p2
    assert p.x == 3
    assert p.y == 4


def test_points_subtraction():
    p1 = Point(1, 1)
    p2 = Point(2, 3)
    p = p1 - p2
    assert p.x == -1
    assert p.y == -2


def test_points_equality():
    p1 = Point(1, 1)
    p2 = Point(1, 1)
    p3 = Point(2, 3)
    assert p1 == p2
    assert p1 != p3


def test_point_scaling():
    p = Point(1, 2)
    p2 = p.scale(3)
    assert p2.x == 3
    assert p2.y == 6


def test_point_length():
    p = Point(3, 4)
    assert p.length() == 5


def test_points_dot_product():
    p = Point(3, 4)
    p2 = Point(-4, 3)
    assert p.dot(p2) == 0
    assert p.dot(p) == 25


def test_points_vector_product():
    p = Point(1, 1)
    p2 = Point(-1, 1)
    assert p.vecdot(p) == 0
    assert p.vecdot(p2) == 2
    assert p.vecdot(p2) == -p2.vecdot(p)


def test_point_classify():
    start = Point(0, 0)
    stop = Point(3, 0)
    assert Point(1, 1).classify(start, stop) == LEFT
    assert Point(-1, -1).classify(start, stop) == RIGHT
    assert Point(-1, 0).classify(start, stop) == BEHIND
    assert Point(1, 0).classify(start, stop) == BETWEEN
    assert Point(5, 0).classify(start, stop) == BEYOND
