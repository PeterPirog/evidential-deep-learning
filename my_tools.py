
import matplotlib.pyplot as plt

def plot_history_loss_accuracy(history,loss='loss',val_loss='val_loss',acc='acc',val_acc='val_acc',first_sample=0):
    last_epoch=len(history.history[acc])
    epochs=[item for item in range(first_sample, last_epoch)]
    plt.plot(epochs,history.history[acc][first_sample:])
    plt.plot(epochs,history.history[val_acc][first_sample:])
    #plt.legend([item for item in range(first_sample, last_epoch+1)])
    plt.title(f'model accuracy:{acc}')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()

    # loss
    plt.plot(epochs,history.history[loss][first_sample:])
    plt.plot(epochs,history.history[val_loss][first_sample:])
    plt.title(f'model loss:{loss}')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()