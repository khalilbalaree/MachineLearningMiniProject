import numpy as np
import matplotlib.pyplot as plt
from loadData import *
from scipy.special import expit

# SVM
def calculate_cost_gradient(Z, X, Y):
    # Z: N x 1
    # Y: N x 1
    # X: N x d+1
    # w: d+1 x 1
    dW = np.zeros([X.shape[1],1])

    for n in range(X.shape[0]):
        yZ = Y[n] * Z[n]  # 1 x 1
        this_x = X[n]  # 1 x d+1

        dw = np.zeros([X.shape[1],1])

        for i in range(X.shape[1]):
            if yZ < 1:
                dw[i][0] = - Y[n] * this_x[i]
            else:
                dw[i][0] = 0     
        dW += dw

    dW = dW/X.shape[0]  # average
    # print(dW)
    return dW

def predict(X, W, t = None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K
    # t: N x 1
    # Z: N x 1

    Z = np.dot(X, W)
    y = np.sign(Z)
    
    loss = np.average(np.clip(1-Z*t, a_min=0, a_max=99999)) #hinge loss
    correct = (y == t).sum()
    acc = correct / X.shape[0]

    return Z, loss, acc

def train(X_train, y_train, X_val, y_val, alpha):
    N_train = X_train.shape[0]
    N_val   = X_val.shape[0]
   
    # init weight
    w = np.zeros([X_train.shape[1], 1])
    
    # init return val
    epoch_best = 0
    acc_best = 0
    W_best = None
    track_loss_train = []
    track_acc_val = []

    for epoch in range(MaxIter):
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size)) ):  
            X_batch = X_train[b*batch_size : (b+1)*batch_size]
            y_batch = y_train[b*batch_size : (b+1)*batch_size]

            Z, loss, _ = predict(X_batch, w, y_batch)

            loss_this_epoch += loss

            # calculate gradient
            gradient = calculate_cost_gradient(Z, X_batch, y_batch) + decay * w
            decay_alpha = np.power(0.96, epoch) * alpha # learning rate decay
            w = w - decay_alpha * gradient

        avg_loss_this_epoch = loss_this_epoch / (N_train/batch_size)
        _, _, acc = predict(X_val, w, y_val)
        # print('epoch:', epoch)
        # print('avg loss this epoch:', avg_loss_this_epoch)
        # print('valid loss:',loss, '\nvalid accuracy:',acc)
        track_loss_train.append(avg_loss_this_epoch)
        track_acc_val.append(acc)
        if acc > acc_best:
            acc_best = acc
            epoch_best = epoch
            W_best = w 

    return epoch_best, acc_best, W_best, track_loss_train, track_acc_val


#Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = loadSpamData(True)

print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)

N_class = 2

batch_size   = 10    # batch size
MaxIter = 200        # Maximum iteration
decay = 1e-2
epsilon = 1e-12

acc_val = []
alpha_set = np.arange(1e-4, 1e-3, 1e-4)
acc_best = 0
W_best = None
epoch_best = None
loss_curve = None
acc_curve = None
best_alpha = None
for alpha in alpha_set:
    print(alpha)
    ep, acc, w, track_loss_train, track_acc_val = train(X_train, t_train, X_val, t_val, alpha)
    acc_val.append(acc)
    if acc > acc_best:
        best_alpha = alpha
        acc_best = acc
        W_best = w
        epoch_best = ep
        loss_curve = track_loss_train
        acc_curve = track_acc_val

print('Best alpha:',best_alpha,'\nBest epoch:', epoch_best, 'with acc in val:',acc_best)