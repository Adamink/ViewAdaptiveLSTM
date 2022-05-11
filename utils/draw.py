import matplotlib.pyplot as plt
import numpy as np
import sys
import os
if __name__ == "__main__":
    folder = sys.argv[1]
    version = sys.argv[2]
    # loss_train = np.load(os.path.join(folder, 'loss_' + version + '_train.npy'))
    # loss_test = np.load(os.path.join(folder, 'loss_' + version + '_test.npy'))
    acc_train = np.load(os.path.join(folder, 'acc_' + version + '_train.npy'))
    acc_test = np.load(os.path.join(folder, 'acc_' + version + '_test.npy'))
    epochs = range(1, len(acc_train) + 1)
    '''
    plt.figure(0)
    plt.plot(epochs, loss_train, 'r--')
    plt.plot(epochs, loss_test, 'b-')
    plt.legend(['Training ' + 'loss', 'Test ' + 'loss'])
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.savefig(os.path.join(folder, 'loss_' + version + '.png'))
    plt.close(0)
    '''
    plt.figure(1)
    plt.plot(epochs, acc_train, 'r--')
    plt.plot(epochs, acc_test, 'b-')
    plt.legend(['Training ' + 'acc', 'Test ' + 'acc'])
    plt.xlabel('Epoch')
    plt.ylabel('acc')
    plt.savefig(os.path.join(folder, 'acc' + version + '.png'))
    plt.close(1)



