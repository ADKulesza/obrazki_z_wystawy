import numpy as np


def average_tf(filename, x_shape=10, y_shape=100):
    """
    Gets average time-frequency map.
    """
    data = np.load(filename)

    f_shape = data.shape[3] // x_shape
    t_shape = data.shape[4] // y_shape
    average_data = np.zeros((data.shape[0], 19, 2,
                             f_shape,
                             t_shape))

    for x in range(0, f_shape):
        for y in range(0, t_shape):
            average_data[:, :, :, x, y] = np.mean(data[:, :, :,
                                                  x * x_shape:(x + 1) * x_shape,
                                                  y * y_shape:(y + 1) * y_shape], axis=(3, 4))

    return average_data
