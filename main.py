import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import plotly.express as px

M = np.array([[4 / 5, -3 / 5], [3 / 5, 4 / 5]])


def make_plot(x_from, x_to, n_points, alpha, beta, rotation_power=0):
    def make_coefficient(i: int, j: int) -> float:
        u = 50 * (i / np.pi - np.floor(i / np.pi))
        v = 50 * (j / np.pi - np.floor(j / np.pi))
        w = u * v * (u + v)
        return 2 * (w - np.floor(w)) - 1

    def height_coefficients(x_from, x_to):
        flat_array = np.array(
            [
                make_coefficient(ii, jj)
                for ii in range(x_from, x_to + 1)
                for jj in range(x_from, x_to + 1)
            ]
        )
        return flat_array.reshape((x_to - x_from + 1, x_to - x_from + 1))

    x = np.linspace(x_from, x_to, n_points)
    z = x
    p = np.array([x, z])
    if rotation_power > 0:
        rotated_p = 2**rotation_power * np.matmul(M**rotation_power, p)
        x = rotated_p[0].reshape(
            n_points,
        )
        x = np.clip(x, x_from, x_to)
        z = rotated_p[1].reshape(
            n_points,
        )
        z = np.clip(z, x_from, x_to)
    i = np.floor(x).astype(int)
    j = np.floor(z).astype(int)

    X, Z = np.meshgrid(x, z)

    a = height_coefficients(x_from, x_to)

    b = np.zeros_like(a)
    c = np.zeros_like(a)
    d = np.zeros_like(a)
    b[0:-1, :] = a[1:, :]  # b[i-1,j] = a[i,j]
    c[:, 0:-1] = a[:, 1:]  # c[i,j-1] = a[i,j]
    d[0:-1, 0:-1] = a[1:, 1:]  # d[i-1,j-1] = a[i,j]

    def S(x: float, alpha: float, beta: float) -> float:

        lamda = np.clip(np.clip((x - alpha) / (beta - alpha), 0, np.inf), -np.inf, 1)
        return lamda * lamda * (3 - 2 * lamda)

    first = (
        a[i, j]
        + (b[i, j] - a[i, j]) * S(X - i, alpha, beta)
        + (c[i, j] - a[i, j]) * S(Z - j, alpha, beta)
        + (a[i, j] - b[i, j] - c[i, j] + d[i, j])
        * S(X - i, alpha, beta)
        * S(Z - j, alpha, beta)
    )
    return X, Z, (0.5) ** rotation_power * first


def graph_update(x_from, x_to, n_points, alpha, beta, rotations):
    X, Z, y = make_plot(x_from, x_to, n_points, alpha, beta, 0)
    if rotations > 0:
        for n in range(2, rotations):
            _, _, y1 = make_plot(x_from, x_to, n_points, alpha, beta, 1)
            y += y1
    fig = go.Figure(data=[go.Surface(z=y)])
    return fig


if __name__ == "__main__":
    # vectorized
    n_points = 10
    x_from = -1
    x_to = 20
    alpha = -5
    beta = 3
    fig = graph_update(x_from, x_to, n_points, alpha, beta, 4)
    fig.show()
