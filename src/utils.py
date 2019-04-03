from keras import callbacks

def callback_for_training(tf_log_dir_name='./log/',patience_lr=10,snapshot_name=None):
    cb = [None] * 3
    """
    Tensorboard log callback
    """
    tb = callbacks.TensorBoard(log_dir=tf_log_dir_name, histogram_freq=0)
    cb[0]= tb
   
    
    """
    Early Stopping callback
    """
    #Uncomment for usage
    # early_stop = callbacks.EarlyStopping(monitor='val_acc', min_delta=0, patience=5, verbose=1, mode='auto',save_best_only=True)
    # cb.apppend(early_stop)
    
    """
    Model Checkpointer
    """
    if snapshot_name != None:
        checkpointer = callbacks.ModelCheckpoint(filepath="optic-net.{epoch:02d}-{val_acc:.2f}.hdf5",
                                verbose=0,
                                monitor='val_acc')
    else :
        checkpointer = callbacks.ModelCheckpoint(filepath=snapshot_name+".{epoch:02d}-{val_acc:.2f}.hdf5",
                                verbose=0,
                                monitor='val_acc')
    cb[1] = checkpointer
    
    """
    Reduce Learning Rate
    """
    reduce_lr_loss = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=6, verbose=1, min_lr=1e-8, mode='auto')
    cb[2] = reduce_lr_loss
    
    return cb
