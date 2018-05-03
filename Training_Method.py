import numpy as np


def train_model(x_model_1, w_model_1, x_model_2, w_model_2, x_model_3, w_model_3, bias):
    return np.multiply(x_model_1, w_model_1) + np.multiply(x_model_2, w_model_2) \
           + np.multiply(x_model_3, w_model_3) + bias


# Training Function starts here
def propagate(w1, w2, w3, b, x, y, t_refined, a_light, learning_rate):
    m, n = x.shape[0], x.shape[1]
    x1 = np.zeros(x.shape)
    x1[:, :, 0] = np.divide(x[:, :, 0], t_refined)
    x1[:, :, 1] = np.divide(x[:, :, 1], t_refined)
    x1[:, :, 2] = np.divide(x[:, :, 2], t_refined)
    x2 = np.zeros(x.shape)
    x2[:, :, 0] = np.divide(a_light[0], t_refined)
    x2[:, :, 1] = np.divide(a_light[1], t_refined)
    x2[:, :, 2] = np.divide(a_light[2], t_refined)

    # Forward propagation
    y0 = train_model(x1[:, :, 0], w1[0], x2[:, :, 0], w2[0], a_light[0], w3[0], b[0])
    y1 = train_model(x1[:, :, 1], w1[1], x2[:, :, 1], w2[1], a_light[1], w3[1], b[1])
    y2 = train_model(x1[:, :, 2], w1[2], x2[:, :, 2], w2[2], a_light[2], w3[2], b[2])

    # Backward propagation
    dw10 = np.sum(np.multiply(x1[:, :, 0], (y0 - y[:, :, 0])), dtype=np.float64) / (m*n)
    dw11 = np.sum(np.multiply(x1[:, :, 1], (y1 - y[:, :, 1])), dtype=np.float64) / (m*n)
    dw12 = np.sum(np.multiply(x1[:, :, 2], (y2 - y[:, :, 2])), dtype=np.float64) / (m*n)
    dw1 = [dw10, dw11, dw12]

    dw20 = np.sum(np.multiply(x2[:, :, 0], (y0 - y[:, :, 0])), dtype=np.float64) / (m*n)
    dw21 = np.sum(np.multiply(x2[:, :, 1], (y1 - y[:, :, 1])), dtype=np.float64) / (m*n)
    dw22 = np.sum(np.multiply(x2[:, :, 2], (y2 - y[:, :, 2])), dtype=np.float64) / (m*n)
    dw2 = [dw20, dw21, dw22]

    dw30 = np.sum(np.multiply(a_light[0], (y0 - y[:, :, 0])), dtype=np.float64) / (m*n)
    dw31 = np.sum(np.multiply(a_light[1], (y1 - y[:, :, 1])), dtype=np.float64) / (m*n)
    dw32 = np.sum(np.multiply(a_light[2], (y2 - y[:, :, 2])), dtype=np.float64) / (m*n)
    dw3 = [dw30, dw31, dw32]

    db0 = np.sum(y0 - y[:, :, 0], dtype=np.float64)/(m*n)
    db1 = np.sum(y1 - y[:, :, 1], dtype=np.float64)/(m*n)
    db2 = np.sum(y2 - y[:, :, 2], dtype=np.float64)/(m*n)
    db = [db0, db1, db2]

    w1 = w1 - np.multiply(learning_rate, dw1)
    w2 = w2 - np.multiply(learning_rate, dw2)
    w3 = w3 - np.multiply(learning_rate, dw3)
    b = b - np.multiply(learning_rate, db)

    return w1, w2, w3, b
