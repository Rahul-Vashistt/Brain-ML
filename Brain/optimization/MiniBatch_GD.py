import pandas as pd
import numpy as np
from Brain.utils import __lr_schedule

def Mini_Batch_GD(X,y,learning_rate,epochs,batch_size,n_batches):
    thetha = np.ones(X.shape[1]) # (p+1,)
    n = X.shape[0] # number of samples

    try:
        if batch_size is not None and n_batches is None:
            n_batches = int(np.ceil(n/batch_size))
        elif batch_size is None and n_batches is not None:
            batch_size = int(np.ceil(n/n_batches))
        else:
            raise ValueError('Provide One param, either batch_size or n_batches and the value should be an Integer')
    except:
        raise TypeError('Expected only Integers!')
    
    for epoch in range(epochs):
        indices = np.random.permutation(n)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        for batch in range(0,n,batch_size):
            start = batch
            end = batch + batch_size

            x_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            try:
                lr = __lr_schedule(epoch) if learning_rate is None else float(learning_rate)
            except:
                raise TypeError(f'Expected Integers or float But got {learning_rate}')

            y_pred = np.dot(x_batch,thetha)
            residual = y_pred - y_batch
            Gradient_thetha = (2/batch_size) * residual @ x_batch

            thetha -= lr * Gradient_thetha
    
        # Convergence check
        if np.linalg.norm(Gradient_thetha) < 1e-6:
            break
    return thetha