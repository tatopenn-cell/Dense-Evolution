from dense_evolution.protocols.bb84 import bb84_run


def test_perfect_channel_zero_qber():
    qber, n_sifted = bb84_run(500, p_channel=0.0, eve=False, seed=0)
    assert qber == 0.0
    assert n_sifted > 0


def test_depolarizing_channel_matches_two_thirds_p():
    p = 0.10
    expected = 2 * p / 3
    qber, _ = bb84_run(2000, p_channel=p, eve=False, seed=1)
    assert abs(qber - expected) < 0.03


def test_intercept_resend_matches_quarter():
    qber, _ = bb84_run(2000, p_channel=0.0, eve=True, seed=2)
    assert abs(qber - 0.25) < 0.03


def test_sifting_keeps_roughly_half_the_rounds():
    _, n_sifted = bb84_run(2000, p_channel=0.0, eve=False, seed=3)
    assert 800 < n_sifted < 1200
