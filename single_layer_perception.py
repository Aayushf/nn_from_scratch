import numpy as np
from mnist import MNIST

mndata = MNIST('/home/aayush/projects/nn_from_scratch/')
images, labels = mndata.load_training()
imgs = np.asarray(images, np.float32).reshape(60000, 784)
anss = np.asarray(labels, np.int).reshape(60000)
bsize = 1024
ip = imgs[0:bsize, :] / 255
w1 = np.random.random_sample((784, 10))
y = np.zeros((bsize, 10))
int_list = anss[0:bsize]
for i in range(bsize):
    y[i, int_list[i]] = 1


def softmax(ip):
    maxes = np.max(ip, 1)
    maxes = np.tile(maxes, 10).reshape((10, bsize))
    ip_shift = (ip.T - maxes).T
    exp = np.exp(ip_shift)
    sum = np.sum(exp, 1)
    sum = np.tile(sum, 10)
    sum = np.reshape(sum, (10, bsize))
    softmax = exp.T / sum
    return softmax.T


def error(preds, targets):
    omp = 1 - preds
    omt = 1 - targets
    lomp = np.log(omp)
    lp = np.log(preds)
    return -1 * ((targets * lp) + (omt * lomp))


def full_back(preds, targets):
    return preds - targets


def get_accuracy(preds, targets):
    pred_ints = np.argmax(preds, 1)
    target_ints = np.argmax(targets, 1)
    print(np.mean(np.equal(pred_ints, target_ints).astype(np.float32)))


while True:
    op = np.dot(ip, w1)
    op = softmax(op)
    err = error(op, y)
    print(np.sum(err))
    get_accuracy(op, y)
    t_b = full_back(op, y)
    grad = np.dot(ip.T, t_b)
    w1 = w1 - grad * 0.000001
