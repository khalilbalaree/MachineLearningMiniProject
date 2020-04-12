import numpy as np
import matplotlib.pyplot as plt
from loadData import *
from scipy.special import expit

# neural network
class neuralNetwork:
    # one input layer, one hidden layer and one output layer structure
    def __init__(self, alpha, batch_size, decay):        
        self.w1 = np.random.normal(0.0, pow(58, -0.5), (20, 58))
        self.w2 = np.random.normal(0.0, pow(20, -0.5), (2, 20))

        self.lr = alpha
        self.batch_size = batch_size
        self.decay = decay

        self.X = None
        self.ho = None
        self.fi = None
        self.fo = None

    def forward(self, X):
        self.X = X  #58 x 10
        hi = np.dot(self.w1, X)
        self.ho = expit(hi) #20 x 10
        self.fi = np.dot(self.w2, self.ho) # 2 x 10
        self.fo = expit(self.fi)

    def backward(self, t, epoch):
        t_hat = self.one_hot(t, 2)
        loss = self.cross_entropy(self.fo, t_hat)

        dw2 = np.dot(self.fo-t_hat, self.ho.transpose())/self.batch_size + self.decay * self.w2 #20*2
        djdho = np.dot(np.transpose(self.w2) , self.fo-t_hat)
        dhodhi = self.ho * (1.0 - self.ho)
        djdhi = djdho * dhodhi
        dw1 = np.dot(djdhi, np.transpose(self.X))/self.batch_size  + self.decay * self.w1

        decay_alpha = np.power(0.996, epoch/10) * self.lr
        self.w2 -= decay_alpha * dw2
        self.w1 -= decay_alpha * dw1

        return loss
        
    def predict(self, X, t):
        self.forward(X)
        t_hat = self.one_hot(t, 2)
        loss = self.cross_entropy(self.fo, t_hat)

        predicted = np.argmax(self.fo, 0).reshape(X.shape[1],1)
        correct = (predicted == t).sum()
        acc = correct / X.shape[1]
        
        return loss, acc

    def cross_entropy(self, p, y):
        m = y.shape[1]
        log_likelihood = -y * np.log(p+epsilon)
        loss = np.sum(log_likelihood) / m
        return loss

    def one_hot(self,data, N_class):
        shape = (N_class, data.shape[0])
        one_hot = np.zeros(shape)
        rows = np.arange(data.shape[0])
        one_hot[data[:,0], rows] = 1
        return one_hot



def train(X_train, y_train, X_val, y_val, alpha):
    N_train = X_train.shape[0]
    N_val   = X_val.shape[0]
    
    # init return val
    epoch_best = 0
    acc_best = 0
    nn_best = None
    track_loss_train = []
    track_acc_val = []

    nn = neuralNetwork(alpha, batch_size, decay)

    for epoch in range(MaxIter):
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size)) ):  
            X_batch = X_train[b*batch_size : (b+1)*batch_size]
            y_batch = y_train[b*batch_size : (b+1)*batch_size]

            nn.forward(np.transpose(X_batch))
            loss = nn.backward(y_batch, epoch)

            loss_this_epoch += loss

        avg_loss_this_epoch = loss_this_epoch / (N_train/batch_size)
        loss, acc = nn.predict(np.transpose(X_val), y_val)
        # print('epoch:', epoch)
        # print('avg loss this epoch:', avg_loss_this_epoch)
        # print('valid loss:',loss, '\nvalid accuracy:',acc)
        track_loss_train.append(avg_loss_this_epoch)
        track_acc_val.append(acc)
        if acc > acc_best:
            acc_best = acc
            epoch_best = epoch
            nn_best = nn

    return epoch_best, acc_best, nn_best, track_loss_train, track_acc_val


#Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = loadSpamData()

print(X_train.shape, t_train.shape, X_val.shape, t_val.shape, X_test.shape, t_test.shape)

N_class = 2

batch_size   = 100    # batch size
MaxIter = 10000       # Maximum iteration
decay = 1e-2
epsilon = 1e-12

acc_val = []
alpha_set = np.arange(1e-3, 1e-2, 1e-3)
acc_best = 0
nn_best = None
epoch_best = None
loss_curve = None
acc_curve = None
best_alpha = None
for alpha in alpha_set:
    print(alpha)
    ep, acc, nn, track_loss_train, track_acc_val = train(X_train, t_train, X_val, t_val, alpha)
    acc_val.append(acc)
    if acc > acc_best:
        best_alpha = alpha
        acc_best = acc
        epoch_best = ep
        nn_best = nn
        loss_curve = track_loss_train
        acc_curve = track_acc_val

print('Best alpha:',best_alpha,'\nBest epoch:', epoch_best, 'with acc in val:',acc_best)