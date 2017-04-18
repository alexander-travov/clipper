from clipper import Point

def test_point_addition():
    p1 = Point(1, 1)
    p2 = Point(2, 3)
    p = p1 + p2
    assert p.x == 3
    assert p.y == 4
