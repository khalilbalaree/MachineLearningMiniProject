import csv
import numpy as np

def loadSpamData(pnmode=False):
    data = []
    with open('spambase/spambase.data','r') as f:
        reader = csv.reader(f)
        next(reader, None)
        for line in reader:
            data.append(line)
    X = np.array([x[:-1] for x in data]).astype(np.float)
    y = np.array([x[-1] for x in data]).astype(np.int16)
    
    X = np.concatenate( ( np.ones([X.shape[0],1]), X), axis=1)
    y = y.reshape([-1,1])

    if pnmode:
        y = np.where(y > 0, 1, -1)

    np.random.seed(100)
    np.random.shuffle(X)
    np.random.seed(100)
    np.random.shuffle(y)

    # nearly 3:1:1
    X_train = X[:2800]
    y_train = y[:2800]

    X_val   = X[2800:3700]
    y_val   = y[2800:3700]

    X_test = X[3700:]
    y_test = y[3700:]

    return X_train, y_train, X_val, y_val, X_test, y_test
