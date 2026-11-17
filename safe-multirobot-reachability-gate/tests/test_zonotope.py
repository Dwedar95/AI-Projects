import numpy as np
import pytest
from scipy.spatial import ConvexHull

from reach_gate.zonotope import Zonotope, convex_hull_enclosure


def random_zonotope(rng, dim=3, gens=7, scale=1.0):
    return Zonotope(rng.normal(0, 2, dim), scale * rng.normal(0, 1, (dim, gens)))


def test_box_constructor_and_interval_hull():
    z = Zonotope.from_box([-1, 2, 0], [3, 4, 0.5])
    lo, hi = z.interval_hull()
    assert np.allclose(lo, [-1, 2, 0]) and np.allclose(hi, [3, 4, 0.5])
    assert z.dim == 3 and z.num_generators == 3 and z.order == 1.0
    with pytest.raises(ValueError):
        Zonotope.from_box([1.0], [0.0])
    with pytest.raises(ValueError):
        Zonotope(np.zeros(2), np.zeros((3, 1)))


def test_linear_map_and_minkowski_sum_match_support_function():
    rng = np.random.default_rng(0)
    for _ in range(20):
        z1, z2 = random_zonotope(rng), random_zonotope(rng)
        m = rng.normal(0, 1, (3, 3))
        d = rng.normal(0, 1, 3)
        # Support of a linear image: h_{MZ}(d) = h_Z(M^T d).
        assert np.isclose(z1.linear_map(m).support(d), z1.support(m.T @ d))
        # Support functions add under Minkowski sums.
        assert np.isclose((z1 + z2).support(d), z1.support(d) + z2.support(d))


def test_samples_lie_inside_interval_hull_and_set():
    rng = np.random.default_rng(1)
    z = random_zonotope(rng, dim=3, gens=9)
    lo, hi = z.interval_hull()
    for p in np.vstack([z.sample(rng, 30), z.sample(rng, 30, boundary=True)]):
        assert np.all(p >= lo - 1e-9) and np.all(p <= hi + 1e-9)
        assert z.contains_point(p)


def test_contains_point_rejects_points_outside_but_inside_the_hull():
    # A thin diagonal segment: its interval hull is the unit square.
    z = Zonotope(np.zeros(2), np.array([[1.0], [1.0]]))
    assert z.contains_point([0.5, 0.5])
    assert not z.contains_point([0.5, -0.5])
    assert not z.contains_point([2.0, 2.0])
    assert Zonotope.point([1.0, 2.0]).contains_point([1.0, 2.0])


def test_order_reduction_is_an_over_approximation():
    rng = np.random.default_rng(2)
    for _ in range(10):
        z = random_zonotope(rng, dim=3, gens=40)
        r = z.reduce_order(4.0)
        assert r.num_generators <= 12
        for d in rng.normal(0, 1, (50, 3)):
            assert r.support(d) >= z.support(d) - 1e-9
        for p in z.sample(rng, 10, boundary=True):
            assert r.contains_point(p)
    small = random_zonotope(rng, dim=3, gens=5)
    assert small.reduce_order(4.0) is small


def test_zonotope_intersection_is_exact_where_interval_hulls_overlap():
    diag = Zonotope(np.zeros(2), np.array([[1.0], [1.0]]))        # segment from (-1,-1) to (1,1)
    box_off = Zonotope.from_box([0.4, -0.9], [0.9, -0.4])         # inside the hull, off the segment
    box_on = Zonotope.from_box([0.4, 0.3], [0.9, 0.8])
    assert not diag.intersects(box_off)
    assert diag.intersects(box_on)
    assert diag.intersects_box([0.4, 0.3], [0.9, 0.8])
    assert not diag.intersects_box([3.0, 3.0], [4.0, 4.0])


def test_intersection_agrees_with_sampling_on_random_pairs():
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(40):
        z1 = random_zonotope(rng, dim=2, gens=4, scale=0.4)
        z2 = random_zonotope(rng, dim=2, gens=4, scale=0.4)
        result = z1.intersects(z2)
        assert result == z2.intersects(z1)
        # Any sampled point of z1 that lies in z2 proves an intersection.
        if any(z2.contains_point(p) for p in z1.sample(rng, 40)):
            assert result
            hits += 1
    assert hits > 0


def test_inside_box():
    z = Zonotope(np.array([1.0, 1.0]), np.array([[0.3, 0.1], [0.0, 0.2]]))
    assert z.inside_box([0.5, 0.7], [1.5, 1.3])
    assert not z.inside_box([0.7, 0.7], [1.5, 1.3])


def test_convex_hull_enclosure_contains_all_interpolations():
    rng = np.random.default_rng(4)
    z1 = random_zonotope(rng, dim=3, gens=4)
    z2 = random_zonotope(rng, dim=3, gens=6)
    hull = convex_hull_enclosure(z1, z2)
    for a, b in zip(z1.sample(rng, 25, boundary=True), z2.sample(rng, 25, boundary=True),
                    strict=True):
        for lam in (0.0, 0.3, 0.7, 1.0):
            assert hull.contains_point(lam * a + (1 - lam) * b)


def test_projection_and_cartesian_product():
    z = Zonotope.from_box([0, 0], [2, 4]).cartesian_product(Zonotope.from_box([1], [3]))
    lo, hi = z.interval_hull()
    assert np.allclose(lo, [0, 0, 1]) and np.allclose(hi, [2, 4, 3])
    lo2, hi2 = z.project([0, 2]).interval_hull()
    assert np.allclose(lo2, [0, 1]) and np.allclose(hi2, [2, 3])


def test_vertices_2d_match_convex_hull_of_extreme_points():
    rng = np.random.default_rng(5)
    z = random_zonotope(rng, dim=2, gens=5)
    verts = z.vertices_2d()
    assert len(verts) == 10
    signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * 5)).reshape(5, -1).T
    extreme = z.center + signs @ z.generators.T
    assert np.isclose(ConvexHull(verts).volume, ConvexHull(extreme).volume)
