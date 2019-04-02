from keras.preprocessing.image import ImageDataGenerator

def Kermany2018(batch_size,image_size,data_dir):

    '''
    Publication : https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
    Dataset : https://data.mendeley.com/datasets/rscbjbr9sj/3
    '''
    train_size = 83484
    test_size = 1000
    classes=4

    train_path = data_dir+'/train'
    test_path = data_dir+'/test'

    train_batches = train_datagen.flow_from_directory(train_path, target_size=(image_size,image_size),color_mode='rgb', classes=classes, batch_size=batch_size,class_mode='categorical')
    test_batches = test_datagen.flow_from_directory(test_path, target_size=(image_size,image_size),color_mode='rgb', classes=classes, batch_size=batch_size, class_mode='categorical')

    return train_batches, test_batches


def Srinivasan2014(batch_size,image_size,data_dir):

    return 
