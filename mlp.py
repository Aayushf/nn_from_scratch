import numpy as np
from mnist import MNIST
import os
from time import sleep
import random

mndata = MNIST('/home/aayush/projects/nn_from_scratch/')
images, labels = mndata.load_training()
it, lt = mndata.load_testing()
imgs = np.asarray(images, np.float32).reshape(60000, 784)
anss = np.asarray(labels, np.int).reshape(60000)
imgst = np.asarray(it, np.float32).reshape(10000, 784)
ansst = np.asarray(lt, np.int).reshape(10000)


def softmax(ip):
    bsize = ip.shape[0]
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
    return np.mean(np.equal(pred_ints, target_ints).astype(np.float32))


layer_sizes = [784, 256, 32, 10]
weights = []
layers = []
backprop_state = []
grads = []


def initialize_weights():
    for i in range(len(layer_sizes) - 1):
        weights.append(np.random.normal(0, 1, (layer_sizes[i], layer_sizes[i + 1])) / 100)
        print(weights[-1].shape)


def forward_pass(ip):
    layers.clear()
    layers.append(ip)
    i = 1
    for w in weights:
        layers.append(np.dot(layers[-1], w))
        # print('LAYER' + str(i))
        # print(layers[-1])
        i += 1
    layers.append(softmax(layers[-1]))


def back_prop(preds, targets, lr):
    bsize = targets.shape[0]
    grads.clear()
    back_temp = preds - targets
    for i in range(len(layers) - 3, -1, -1):
        grads.append(np.dot(layers[i].T, back_temp) / bsize)
        back_temp = np.dot(back_temp, weights[i].T)
    grads.reverse()
    for t in range(len(grads)):
        weights[t] = weights[t] - grads[t] * lr


def get_batch_train(bsize):
    r = random.randint(0, 60000 - 1 - bsize)
    ip = imgs[r:r + bsize, :] / 255
    y = np.zeros((bsize, 10))
    int_list = anss[r:r + bsize]
    for i in range(bsize):
        y[i, int_list[i]] = 1
    return ip, y


def get_batch_test(bsize):
    r = random.randint(0, 10000 - 1 - bsize)
    ip = imgst[r:r + bsize, :] / 255
    y = np.zeros((bsize, 10))
    int_list = ansst[r:r + bsize]
    for i in range(bsize):
        y[i, int_list[i]] = 1
    return ip, y


initialize_weights()
lr = 0.1
q = 0
prev_error = 0
testacc = 0
while True:
    q += 1
    ip, y = get_batch_train(64)
    forward_pass(ip)
    os.system("clear")
    print("Iteration", q)
    c_err = np.sum(error(layers[-1], y))
    print("Loss: ", c_err)
    print("Loss Delta: ", c_err - prev_error)
    print("Train Accuracy: ", get_accuracy(layers[-1], y))
    print("Test Accuracy: ", testacc)
    prev_error = c_err
    lr/=1.00001
    back_prop(layers[-1], y, lr)
    if q % 10 == 0:
        i, a = get_batch_test(8000)
        forward_pass(i)
        testacc = get_accuracy(layers[-1], a)
    sleep(0.03)
