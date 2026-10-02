import numpy as np

from rwe.eval_sts import cosine_pairs, eval_pairs, rankdata, spearman


def test_rankdata_average_ties():
    assert rankdata([10, 20, 20, 30]).tolist() == [1.0, 2.5, 2.5, 4.0]


def test_spearman_perfect_reverse_and_ties():
    assert abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-12
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12
    assert abs(spearman([1, 2, 2, 3], [1, 2, 3, 4]) - 4.5 / np.sqrt(4.5 * 5)) < 1e-9


def test_spearman_constant_input_is_zero():
    assert spearman([1, 1, 1], [1, 2, 3]) == 0.0


def test_cosine_pairs_ignores_scale():
    a = np.array([[1.0, 0.0], [1.0, 1.0]])
    b = np.array([[5.0, 0.0], [-1.0, -1.0]])
    assert np.allclose(cosine_pairs(a, b), [1.0, -1.0])


def test_eval_pairs_with_fixed_encoder():
    table = {"a": [1, 0], "b": [1, 0.1], "c": [0, 1], "d": [1, 1]}
    enc = lambda texts: np.array([table[t] for t in texts], dtype=np.float32)
    # pair cosines: (a,b) ~ high, (a,c) = 0, (a,d) ~ 0.707
    score = eval_pairs(enc, ["a", "a", "a"], ["b", "c", "d"], np.array([0.9, 0.1, 0.5]))
    assert abs(score - 1.0) < 1e-9  # cosine rank order b > d > c matches gold
    assert eval_pairs(enc, ["a", "a", "a"], ["b", "d", "c"], np.array([3.0, 2.0, 1.0])) == 1.0
