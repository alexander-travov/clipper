import numpy as np
from clipper import *


def test_polygon_vertex():
    p = Polygon(np.array([1., 2., 3.]), np.array([0., 0., 1.]))
    assert p.N == 3
    assert p.v(0) == Point(1., 0.)
    assert p.v(1) == Point(2., 0.)
    assert p.v(2) == Point(3., 1.)
    assert p.v(3) == Point(1., 0.)
    assert p.v(4) == Point(2., 0.)
    assert p.v(5) == Point(3., 1.)


def test_polygon_edge():
    p = Polygon(np.array([1., 2., 3.]), np.array([0., 0., 1.]))
    assert p.e(0) == Edge(Point(1., 0.), Point(2., 0.))
    assert p.e(1) == Edge(Point(2., 0.), Point(3., 1.))
    assert p.e(2) == Edge(Point(3., 1.), Point(1., 0.))
    assert p.e(3) == Edge(Point(1., 0.), Point(2., 0.))


def test_polygon_contains():
    p = Polygon(np.array([0., 0., 1., 1.]), np.array([0., 1., 1., 0.]))
    assert p.contains(Point(.5, .5))
    assert not p.contains(Point(1.5, .5))


def test_polygon_area():
    p = Polygon(np.array([0., 0., 1., 1.]), np.array([0., 1., 1., 0.]))
    assert p.area() == 1.


def test_polygon_intersection_area():
    p = Polygon(np.array([0., 0., 1., 1.]), np.array([0., 1., 1., 0.]))
    p2 = Polygon(np.array([-2., -2., 2., 2.]), np.array([-2., 2., 2., -2.]))
    p3 = Polygon(np.array([0., -3., 0., 3.]), np.array([-3., 0., 3., 0.]))
    p4 = Polygon(np.array([4., 4., 5., 5.]), np.array([4., 5., 5., 4.]))
    p5 = Polygon(np.array([1., 2., 3., 2.]), np.array([0.5, 1., 1, 0.5]))
    assert intersection_area(p, p2) == 1.
    assert intersection_area(p, p3) == 1.
    assert intersection_area(p3, p) == 1.
    assert intersection_area(p, p4) == 0.
    assert intersection_area(p2, p3) == 14
    assert intersection_area(p2, p4) == 0
    assert intersection_area(p3, p4) == 0
    # assert intersection_area(p, p5) == 0
