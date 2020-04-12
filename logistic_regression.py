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

def train(X_train, y_train, X_val, y_val):
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
        print('epoch:', epoch)
        print('avg loss this epoch:', avg_loss_this_epoch)
        print('valid loss:',loss, '\nvalid accuracy:',acc)
        track_loss_train.append(avg_loss_this_epoch)
        track_acc_val.append(acc)
        if acc > acc_best:
            acc_best = acc
            epoch_best = epoch
            W_best = w 

    return epoch_best, acc_best, W_best, track_loss_train, track_acc_val


X_train, y_train, X_val, y_val, X_test, y_test = loadSpamData()
print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

N_class = 2

alpha   = 3e-4          # learning rate
batch_size  = 10        # batch size
MaxIter = 1000          # Maximum iteration
decay = 0.01            # weight decay
epsilon = 1e-12         # avoid error message

epoch_best, acc_best, W_best, track_loss_train, track_acc_val = train(X_train, y_train, X_val, y_val)

_, _, _, acc_test = predict(X_test, W_best, y_test)

print('\nAt epoch {}.\nvalidation accuracy: {:.2f}%.\ntest accuracy: {:.2f}%'.format(epoch_best, acc_best*100, acc_test*100))
plt.figure(1)
plt.plot(track_loss_train)
plt.xlabel("epochs")
plt.ylabel("training loss")
plt.figure(2)
plt.plot(track_acc_val)
plt.xlabel("epochs")
plt.ylabel("validation acc")
plt.show()