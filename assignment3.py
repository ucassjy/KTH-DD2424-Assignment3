import numpy as np
import scipy.io as scio
import matplotlib.pylab as plt

BN = True
epsilon = 1e-4
rho = 0.9
batch_size = 100
epoch_num = 10
decay_rate = 0.95
node_num = [3072, 50, 30, 10]
W = list()
b = list()
_ReLu = True # If it's false then it'll use sigmoid as activation

def ReadAndReshape(dataFile):
    """ The file is given by .mat format. """
    raw_data = scio.loadmat(dataFile)
    data = raw_data['data']
    data = data.astype(np.float64)
    data = data / 255
    raw_label = raw_data['labels']
    label = list()
    for value in raw_label:
        l = np.zeros(10)
        l[value] = 1
        label.append(l)
    label = np.array(label)
    return data, label

def BatchNormalize(S):
    """ Prevent from vanishing or exploding gradients. """
    mu = np.mean(S, axis=0)
    v = np.mean((S-mu)**2, axis=0)
    S = (S - mu) / np.sqrt(v + epsilon)
    return S

def EvaluateLayer(X, W, b):
    """ Evaluate the output S for every layer. """
    S = [(np.dot(W, x) + b) for x in X]
    S_hat = BatchNormalize(S) if BN else S
    H = np.maximum(S_hat, 0) if _ReLu else 1 / (1 + np.exp(-S_hat))
    return H, S

def EvaluateClassifier(X, W, b):
    """ Evaluate the """
    P = list()
    H = list()
    S = list()
    for i in range(len(W) - 1):
        if i is 0:
            h = X
        h, s = EvaluateLayer(h, W[i], b[i])
        H.append(h)
        S.append(s)
    w = W[-1]
    bi = b[-1]
    for hi in h:
        s = np.dot(w, hi) + bi
        p = np.exp(s) / np.sum(np.exp(s))
        P.append(p)
    return P, H, S

def ComputeCost(Y, W, P, my_lambda):
    """ Compute the cross-entrophy loss on training data. """
    l = [np.log(P[i][np.argmax(Y[i])]) for i in range(len(Y))]
    l = -np.mean(l)
    J = l
    for w in W:
        J += my_lambda * (w**2).sum()
    return J, l

def ComputeAccuracy(Y, P):
    """ Compute the accuracy of the result. """
    n = 0
    y = len(Y)
    for i in range(y):
        if np.argmax(P[i]) == np.argmax(Y[i]):
            n += 1
    acc = n / y
    return acc

def BatchNormBack(G, s):
    """ Compute backward gradients under batch normalization. """
    mu = np.mean(s, axis=0)
    v = 1 / np.sqrt(np.mean((s-mu)**2, axis=0))
    m_1 = np.mean(G * (s - mu), axis=0)
    m_2 = np.mean(G, axis=0)
    G1 = [G[i] * v - m_1 * (v**3) * (s[i]-mu) - m_2 * v for i in range(len(G))]
    return G1

def ComputeGradients(X, Y, W, b, my_lambda):
    """ Compute backward gradients. """
    grad_W = list()
    grad_b = list()
    P, H, S = EvaluateClassifier(X, W, b)
    G = list()
    G1 = list()
    w = len(W)
    for i in range(w - 1):
        G = P - Y if i is 0 else G1[:]
        h = H[w-i-2]
        x = len(h)
        grad_W1 = [np.dot(G[j].reshape(-1, 1), h[j].reshape(1, -1)) for j in range(x)]
        grad_W1 = np.mean(grad_W1, axis=0) + 2 * my_lambda * W[w-i-1]
        grad_W.append(grad_W1)
        grad_b1 = np.mean(G, axis=0)
        grad_b.append(grad_b1)
        if _RuLu:
            for j in range(x):
                h[j][h[j]>0] = 1
            G1 = [np.dot(g, W[w-i-1]) for g in G]
            G1 = G1 * h
        else:
            G1 = [np.dot(g, W[w-i-1]) for g in G]
            G1 = G1 * h * (1 - h)
        if BN:
            s = S[w-i-2]
            G1 = BatchNormBack(G1, s)
    grad_W1 = [np.dot(G1[i].reshape(-1, 1), X[i].reshape(1, -1)) for i in range(x)]
    grad_W1 = np.mean(grad_W1, axis=0) + 2 * my_lambda * W[0]
    grad_b1 = np.mean(G1, axis=0)
    grad_W.append(grad_W1)
    grad_b.append(grad_b1)
    return grad_W[::-1], grad_b[::-1]

def ComputeGradientsTest(X, Y, W, b, my_lambda):
    """ Test if the function ComputeGradients() is right by using math defination of gradients.
        This function is really slow. """
    grad_W = [np.zeros(w.shape) for w in W]
    grad_b = [np.zeros(bi.shape) for bi in b]
    P, _, _ = EvaluateClassifier(X, W, b)
    cost, _ = ComputeCost(Y, W, P, my_lambda)
    for k in range(len(W)):

        w = W[k]
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                W[k][i][j] += epsilon
                P1, _, _ = EvaluateClassifier(X, W, b)
                cost1, _ = ComputeCost(Y, W, P1, my_lambda)
                W[k][i][j] -= epsilon
                grad_W[k][i][j] = (cost1 - cost) / epsilon

        for i in range(b[k].shape[0]):
            b[k][i] += epsilon
            P1, _, _ = EvaluateClassifier(X, W, b)
            cost1, _ = ComputeCost(Y, W, P1, my_lambda)
            b[k][i] -= epsilon
            grad_b[k][i] = (cost1 - cost) / epsilon

    return grad_W, grad_b

trainFile = "../cifar-10-batches-mat/data_batch_1.mat"
validFile = "../cifar-10-batches-mat/data_batch_2.mat"

data, labels = ReadAndReshape(trainFile)
vdata, vlabels = ReadAndReshape(validFile)

# If train and valid with all 50000 images.
#t_data, t_labels = ReadAndReshape("../cifar-10-batches-mat/data_batch_1.mat")
#data = t_data[:9800]
#labels = t_labels[:9800]
#vdata = t_data[9800:]
#vlabels = t_labels[9800:]

#t_data, t_labels = ReadAndReshape("../cifar-10-batches-mat/data_batch_2.mat")
#data = np.concatenate((data, t_data[:9800]), axis=0)
#labels = np.concatenate((labels, t_labels[:9800]), axis=0)
#vdata = np.concatenate((vdata, t_data[9800:]), axis=0)
#vlabels = np.concatenate((vlabels, t_labels[9800:]), axis=0)

#t_data, t_labels = ReadAndReshape("../cifar-10-batches-mat/data_batch_3.mat")
#data = np.concatenate((data, t_data[:9800]), axis=0)
#labels = np.concatenate((labels, t_labels[:9800]), axis=0)
#vdata = np.concatenate((vdata, t_data[9800:]), axis=0)
#vlabels = np.concatenate((vlabels, t_labels[9800:]), axis=0)

#t_data, t_labels = ReadAndReshape("../cifar-10-batches-mat/data_batch_4.mat")
#data = np.concatenate((data, t_data[:9800]), axis=0)
#labels = np.concatenate((labels, t_labels[:9800]), axis=0)
#vdata = np.concatenate((vdata, t_data[9800:]), axis=0)
#vlabels = np.concatenate((vlabels, t_labels[9800:]), axis=0)

#t_data, t_labels = ReadAndReshape("../cifar-10-batches-mat/data_batch_5.mat")
#data = np.concatenate((data, t_data[:9800]), axis=0)
#labels = np.concatenate((labels, t_labels[:9800]), axis=0)
#vdata = np.concatenate((vdata, t_data[9800:]), axis=0)
#vlabels = np.concatenate((vlabels, t_labels[9800:]), axis=0)

data_mean = np.mean(data, axis=1)
data_mean = data_mean.repeat(3072, axis=0)
data_mean = data_mean.reshape(data.shape[0], -1)

data = data - data_mean
vdata = vdata - data_mean[:vdata.shape[0]]

tl = list()
vl = list()
tJ = list()
vJ = list()

m = len(node_num)

# To search for the best lambda and eta.
for my_lambda in np.arange(0.001, 0.003, 0.002):
    for eta in np.arange(0.051, 0.071, 0.02):
        e = eta
        W = [np.random.normal(0, 0.01, (node_num[i+1], node_num[i])) for i in range(m-1)]
        b = [np.random.normal(0, 0.01, node_num[i+1]) for i in range(m-1)]
        vW = [np.zeros(w.shape) for w in W]
        vb = [np.zeros(bi.shape) for bi in b]
        vW = np.array(vW)
        vb = np.array(vb)

        for epoch in range(epoch_num):
            for i in range(len(data) // batch_size):
                data_batch = data[i*batch_size : (i+1)*batch_size]
                label_batch = labels[i*batch_size : (i+1)*batch_size]

                grad_W, grad_b = ComputeGradients(data_batch, label_batch, W, b, my_lambda)
#                grad_W_test, grad_b_test = ComputeGradientsTest(data_batch, label_batch, W, b, my_lambda)
#                for i in range(len(grad_W_test)):
#                    rw = grad_W_test[i] / grad_W[i]
#                    for j in range(rw.shape[0]):
#                        for k in range(rw.shape[1]):
#                            print(rw[j][k], end="  ")
#                        print(" ")
#                    print(" ")
#                    rb = grad_b_test[i] / grad_b[i]
#                    for j in range(rb.shape[0]):
#                        print(rb[j], end="   ")
#                    print("\n")
                grad_W = np.array(grad_W)
                grad_b = np.array(grad_b)
                vW = rho * vW + eta * grad_W
                W -= vW
                vb = rho * vb + eta * grad_b
                b -= vb

            eta = eta * decay_rate

            P, _, _ = EvaluateClassifier(data, W, b)
            cost, loss = ComputeCost(labels, W, P, my_lambda)
            tacc = ComputeAccuracy(labels, P)
            tl.append(loss)
#            tJ.append(cost)
            print(cost, loss, end=" ")

            P, _, _ = EvaluateClassifier(vdata, W, b)
            cost, loss = ComputeCost(vlabels, W, P, my_lambda)
            vacc = ComputeAccuracy(vlabels, P)
            vl.append(loss)
#            vJ.append(cost)
            print('\t', cost, loss)

            print('eta =', e, ', lambda =', my_lambda, ', tacc =', tacc, ', vacc =', vacc)

        plt.figure(1)
        x = np.linspace(0, epoch_num, epoch_num)
        plt.xlabel("Epoch num")
        plt.ylabel("Loss")
        plt.plot(x, tl, label='Training data')
        plt.plot(x, vl, label='Validation data')
        plt.legend()
#        plt.figure(2)
#        plt.xlabel("Epoch num")
#        plt.ylabel("Cost")
#        plt.plot(x, tJ, label='Training data')
#        plt.plot(x, vJ, label='Validation data')
#        plt.legend()

        testFile = "../cifar-10-batches-mat/test_batch.mat"
        tdata, tlabels = ReadAndReshape(testFile)
        tdata = tdata - data_mean[:10000]
        P, _, _ = EvaluateClassifier(tdata, W, b)
        acc = ComputeAccuracy(tlabels, P)
        print("\nFor eta =", e, ", lambda =", my_lambda, ", batch_size =", batch_size, ", epoch_num =", epoch_num, ":")
        print("\tThe accuracy for test_data is: ", acc)

        plt.show()
