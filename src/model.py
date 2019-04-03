import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Multiply,GlobalAveragePooling2D, Add, Dense, Activation, Maximum, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Lambda, UpSampling2D, DepthwiseConv2D, SeparableConv2D
from keras.optimizers import Adam
from keras.initializers import glorot_uniform

def res_conv(X, filters, base, s):
    
    name_base = base + '/branch'
    
    F1, F2, F3 = filters

    ##### Branch1 is the main path and Branch2 is the shortcut path #####
    
    X_shortcut = X
    
    ##### Branch1 #####
    # First component of Branch1 
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_1')(X)
    X= Activation('relu', name=name_base + '1/relu_1')(X)
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1/conv_1', kernel_initializer=glorot_uniform(seed=0))(X)

    # Second component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_2')(X)
    X = Activation('relu', name=name_base + '1/relu_2')(X)
    X = Conv2D(filters=F2, kernel_size=(2,2), strides=(s,s), padding='same', name=name_base + '1/conv_2', kernel_initializer=glorot_uniform(seed=0))(X)
    
    # Third component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_3')(X)
    X = Activation('relu', name=name_base + '1/relu_3')(X)
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1/conv_3', kernel_initializer=glorot_uniform(seed=0))(X)
    
    ##### Branch2 ####
    X_shortcut = BatchNormalization(axis=-1, name=name_base + '2/bn_1')(X_shortcut)
    X_shortcut= Activation('relu', name=name_base + '2/relu_1')(X_shortcut)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding='valid', name=name_base + '2/conv_1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    
    # Final step: Add Branch1 and Branch2
    X = Add(name=base + '/Add')([X, X_shortcut])

    return X
def res_identity(X, filters, base):
    
    name_base = base + '/branch'
    
    F1, F2, F3 = filters

    ##### Branch1 is the main path and Branch2 is the shortcut path #####
    
    X_shortcut = X
    
    ##### Branch1 #####
    # First component of Branch1 
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_1')(X)
    Shortcut= Activation('relu', name=name_base + '1/relu_1')(X)
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1/conv_1', kernel_initializer=glorot_uniform(seed=0))(Shortcut)

    # Second component BranchOut 1
    X1 = BatchNormalization(axis=-1, name=name_base + '1/ConvBn_2')(X)
    X1 = Activation('relu', name=name_base + '1/ConvRelu_2')(X1)
    X1 = Conv2D(filters=F2, kernel_size=(2,2), dilation_rate=(2, 2),strides=(1,1), padding='same', name=name_base + '1/Conv_2', kernel_initializer=glorot_uniform(seed=0))(X1)
    
    # Second component BrancOut 2
    X2 = BatchNormalization(axis=-1, name=name_base + '1/SepBn_2')(X)
    X2 = Activation('relu', name=name_base + '1/SepRelu_2')(X2)
    X2 = SeparableConv2D(filters=F2, kernel_size=(2,2), dilation_rate=(2, 2),strides=(1,1), padding='same', name=name_base + '1/SepConv_2', kernel_initializer=glorot_uniform(seed=0))(X2)
    
    # Second component Add-BranchOut
    X = Add(name=base + '/Add-2branches')([X1, X2])
    
    # Third component of Branch1
    X = BatchNormalization(axis=-1, name=name_base + '1/bn_3')(X)
    X = Activation('relu', name=name_base + '1/relu_3')(X)
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=name_base + '1/conv_3', kernel_initializer=glorot_uniform(seed=0))(X)    
    
    # Final step: Add Branch1 and the original Input itself
    X = Add(name=base + '/Add')([X_shortcut,X])

    return X

def EncoderDecoder(X, name_base):
    X = MaxPooling2D((3,3), strides=(2,2), padding='same', name = name_base + '/Downsample1')(X)
    #X = Conv2D(outgoing_depth, (2,2), strides=(1,1), dilation_rate=(2,2), padding='same', name = name_base + '/DC1', kernel_initializer=glorot_uniform(seed=0))(X)    
    X = UpSampling2D(size=(2, 2),interpolation='bilinear',name = name_base + '/Upsample1')(X)
    X = Activation('sigmoid', name = name_base + '/Activate')(X)
    return X
def RDBI(X, filters, base, number):
    
    for i in range(number):
        X = res_identity(X, filters, base+ '/id_'+str(1+i))
    
    return X
def OpticNet(input_size,num_of_classes):
    input_shape=(input_size, input_size, 3) # Height x Width x Channel
    X_input = Input(input_shape)

    X = Conv2D(64, (7,7), strides=(2,2), padding='same', name ='CONV1', kernel_initializer=glorot_uniform(seed=0))(X_input)
    X = BatchNormalization(axis=-1, name ='BN1')(X)
    X = Activation('relu', name ='RELU1')(X)
    
    X = res_conv(X, [64,64,256], 'RC0', 1)
    
    # MID 1
    
    X1 = EncoderDecoder(X, 'EncoderDecoder1')
    
    X2 = RDBI(X, [32,32,256], 'RDBI1',4)
    
    X = Multiply(name = 'Mutiply1')([X1,X2])
    
    X = Add(name = 'Add1')([X,X1,X2])
    
    X = res_conv(X, [128,128,512], 'RC1', 2)
    
    # MID 2
    
    X1 = EncoderDecoder(X, 'EncoderDecoder2')
    
    X2 = RDBI(X, [64,64,512], 'RDBI2',4)
    
    X = Multiply(name = 'Mutiply2')([X1,X2])
    
    X = Add(name = 'Add2')([X,X1,X2])
    
    X = res_conv(X, [256,256,1024], 'RC2', 2)
    
    # MID 3
    
    X1 = EncoderDecoder(X, 'EncoderDecoder3')
    
    X2 = RDBI(X, [128,128,1024], 'RDBI3',3)
    
    X = Multiply(name = 'Mutiply3')([X1,X2])
    
    X = Add(name = 'Add3')([X,X1,X2])
    
    X = res_conv(X, [512,512,2048], 'RC3', 2)
    
    # MID 4
    
    X1 = EncoderDecoder(X, 'EncoderDecoder4')
    
    X2 = RDBI(X, [256,256,2048], 'RDBI4',3)
    
    X = Multiply(name = 'Mutiply4')([X1,X2])
    
    X = Add(name = 'Add4')([X,X1,X2])
    
    
    X = GlobalAveragePooling2D(name='global_avg_pool')(X)
    X = Dense(256, name='Dense_1')(X)
    X = Dense(num_of_classes, name='Dense_2')(X)
    X = Activation('softmax', name='classifier')(X)
    
    
    model = Model(inputs=X_input, outputs=X, name='')

    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    return model
