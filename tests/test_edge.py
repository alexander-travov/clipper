from clipper import *


def test_edge_direction():
    e = Edge(Point(1, 3), Point(4, 1))
    d = e.direction()
    assert d.x == 3
    assert d.y == -2


def test_edge_normal():
    e = Edge(Point(1, 3), Point(4, 1))
    n = e.normal()
    assert n.dot(e.direction()) == 0
    assert n.x == -2
    assert n.y == -3


def test_edge_point():
    e = Edge(Point(1, 3), Point(4, 1))
    p = e.point(2)
    assert p.x == 7
    assert p.y == -1


def test_edge_point_classify():
    start = Point(0, 0)
    stop = Point(3, 0)
    e = Edge(start, stop)
    assert e.point_classify(Point(1, 1)) == LEFT
    assert e.point_classify(Point(-1, -1)) == RIGHT
    assert e.point_classify(Point(-1, 0)) == BEHIND
    assert e.point_classify(Point(1, 0)) == BETWEEN
    assert e.point_classify(Point(5, 0)) == BEYOND


def test_edge_edge_classify():
    start = Point(0, 0)
    stop = Point(3, 0)
    e = Edge(start, stop)

    assert e.edge_classify(Edge(Point(1, 1), Point(4, 1))) == PARALLEL
    assert e.edge_classify(Edge(Point(1, 0), Point(2, 0))) == COLLINEAR
    assert e.edge_classify(Edge(Point(1, -1), Point(1, 2))) == SKEW_CROSS
    assert e.edge_classify(Edge(Point(1, 4), Point(1, 2))) == SKEW_NO_CROSS


def test_edge_intersection_point():
    e1 = Edge(Point(0, 0), Point(3, 0))
    e2 = Edge(Point(1, -1), Point(3, 1))
    p = e1.intersection_point(e2)
    assert p.x == 2
    assert p.y == 0


def test_edge_aims_at():
    e = Edge(Point(0, 0), Point(3, 0))
    assert Edge(Point(1, -2), Point(2, -1)).aims_at(e)
    assert not Edge(Point(1, -2), Point(2, 1)).aims_at(e)
    assert Edge(Point(1, 2), Point(2, 1)).aims_at(e)
    assert not Edge(Point(1, 2), Point(2, -1)).aims_at(e)
    assert Edge(Point(1, 0), Point(-1, 0)).aims_at(e)
    assert not Edge(Point(1, 0), Point(4, 0)).aims_at(e)
