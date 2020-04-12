import numpy as np
import matplotlib.pyplot as plt
from loadData import *
from scipy.special import expit

# logistic regression
def one_hot(data):
    shape = (data.shape[0], N_class)
    one_hot = np.zeros(shape)
    rows = np.arange(data.shape[0])
    one_hot[rows, data[:,0]] = 1
    return one_hot

def cross_entropy(p, y):
    m = y.shape[0]
    log_likelihood = -y * np.log(p+epsilon)
    loss = np.sum(log_likelihood) / m
    return loss

def predict(X, W, t = None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    Z = np.dot(X, W)
    y = expit(Z)
    t_hat = one_hot(t)
    loss = cross_entropy(y, t_hat)
    
    predicted = np.argmax(y, 1).reshape(X.shape[0],1)
    correct = (predicted == t).sum()
    acc = correct / X.shape[0]

    return y, t_hat, loss, acc

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

            y, t_hat, loss, _ = predict(X_batch, w, y_batch)

            loss_this_epoch += loss

            # calculate gradient
            gradient = np.dot(X_batch.transpose(), (y-t_hat))/batch_size + decay * w
            decay_alpha = np.power(0.99, epoch) * alpha # learning rate decay
            w = w - decay_alpha * gradient
        avg_loss_this_epoch = loss_this_epoch / (N_train/batch_size)
        _, _, loss, acc = predict(X_val, w, y_val)
        track_loss_train.append(avg_loss_this_epoch)
        track_acc_val.append(acc)
        if acc > acc_best:
            acc_best = acc
            epoch_best = epoch
            W_best = w 

    return epoch_best, acc_best, W_best, track_loss_train, track_acc_val

#Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = loadSpamData()

print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)

N_class = 2

batch_size   = 10    # batch size
MaxIter = 1000        # Maximum iteration
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
# plt.figure(1)
# plt.plot(loss_curve)
# plt.xlabel("epochs")
# plt.ylabel("training loss")
# plt.figure(2)
# plt.plot(acc_curve)
# plt.xlabel("epochs")
# plt.ylabel("validation acc")
# plt.show()