
import matplotlib.pyplot as plt


def  plot_loss_acc(history,snapshot_name=None):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(loss))
    plt.plot(epochs, loss, 'bo')
    plt.plot(epochs, val_loss, 'g')
    plt.title('Training and validation loss')
    plt.legend(['train', 'val'], loc='upper right')
    if snapshot_name == None:
        filename= 'OpticNet_loss.png'
    else:
        filename= snapshot_name+'_loss.png'
    plt.savefig(filename)
    plt.show()
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'b')
    plt.plot(epochs, val_acc, 'g')
    plt.title('Training and validation accuracy')
    plt.legend(['train', 'val'], loc='lower right')
    if snapshot_name == None:
        filename= 'OpticNet_acc.png'
    else:
        filename= snapshot_name+'_acc.png'
    plt.savefig(filename)
    plt.show() 