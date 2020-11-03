import pickle
import matplotlib.pyplot as plt
import numpy as np

with open('ssd300_fpn38_history.pkl', 'rb') as f:
    history = pickle.load(f)    
    
    epoch_size = 517
    loss_history = history['loss']


    loss_history = [np.mean(loss_history[i:i+epoch_size]) for i in range(0, 120000-517, 517)]

    plt.plot(loss_history)
    plt.show()