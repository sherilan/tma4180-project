import pytest
import numpy as np

import alg


@pytest.mark.parametrize(
    'seed', [(x*100 + x,) for x in range(100)]
)
def test_objective_flat(seed):
    M = 20
    random = np.random.RandomState(seed)
    A = random.uniform(-10, 10, size=(M, 2))
    v = random.uniform(0.001, 10, size=M)
    x = x1, x2 = random.uniform(-10, 10, size=2)

    f_x = 0
    for i in range(M):
        dist = ((x1 - A[i,0])**2 + (x2 - A[i,1])**2)**0.5
        assert dist > 0, 'Got zero distance in weiszfeld update'
        f_x += v[i] * dist

    f_x_numpy = alg.objective(A, v, x)
    assert f_x_numpy.shape == tuple([])
    assert np.allclose(f_x, f_x_numpy)


@pytest.mark.parametrize(
    'seed', [(x*100 + x,) for x in range(100)]
)
def test_objective_2d(seed):
    M = 20
    N = 10
    random = np.random.RandomState(seed)
    A = random.uniform(-10, 10, size=(M, 2))
    v = random.uniform(0.001, 10, size=M)
    xx = random.uniform(-10, 10, size=(N, 2))

    f_xx = alg.objective(A, v, xx)
    assert f_xx.shape == (N,)

    for j in range(N):
        assert np.all(f_xx[j] == alg.objective(A, v, xx[j]))

@pytest.mark.parametrize(
    'seed', [(x*100 + x,) for x in range(100)]
)
def test_objective_3d(seed):
    M = 20
    N = 10
    random = np.random.RandomState(seed)
    A = random.uniform(-10, 10, size=(M, 2))
    v = random.uniform(0.001, 10, size=M)
    xx = random.uniform(-10, 10, size=(N, N, 2))

    f_xx = alg.objective(A, v, xx)
    assert f_xx.shape == (N,N)

    for j in range(N):
        for k in range(N):
            assert np.all(f_xx[j, k] == alg.objective(A, v, xx[j, k]))


@pytest.mark.parametrize(
    'seed', [(x*100 + x,) for x in range(100)]
)
def test_weizefeld_update(seed):
    M = 20
    random = np.random.RandomState(seed)
    A = random.uniform(-10, 10, size=(M, 2))
    v = random.uniform(0.001, 10, size=M)
    x = x1, x2 = random.uniform(-10, 10, size=2)

    x1_new = 0
    x2_new = 0
    div = 0
    for i in range(M):
        dist = ((x1 - A[i,0])**2 + (x2 - A[i,1])**2)**0.5
        assert dist > 0, 'Got zero distance in weiszfeld update'
        x1_new += v[i] * A[i,0] / dist
        x2_new += v[i] * A[i,1] / dist
        div += v[i] / dist
    x1_new = x1_new / div
    x2_new = x2_new / div

    x_new_manual = np.array([x1_new, x2_new])
    x_new_numpy = alg.weiszfeld_update(A, v, x)

    assert np.allclose(x_new_manual, x_new_numpy)
