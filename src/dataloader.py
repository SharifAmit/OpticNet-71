from keras.preprocessing.image import ImageDataGenerator

def Kermany2018(batch_size,image_size,data_dir):

    '''
    Publication : https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5
    Dataset : https://data.mendeley.com/datasets/rscbjbr9sj/3
    '''
    #train_size = 83484
    #test_size = 1000

    train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   rescale=1.0/255,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_path = data_dir+'/train'
    test_path = data_dir+'/test'

    classes = ['CNV', 'DME','DRUSEN','NORMAL']

    train_batches = train_datagen.flow_from_directory(train_path, target_size=(image_size,image_size),color_mode='rgb', classes=classes, batch_size=batch_size,class_mode='categorical')
    test_batches = test_datagen.flow_from_directory(test_path, target_size=(image_size,image_size),color_mode='rgb', classes=classes, batch_size=batch_size, class_mode='categorical')

    return train_batches, test_batches


def Srinivasan2014(batch_size,image_size,data_dir):

    '''
    Publication : http://www.opticsinfobase.org/boe/abstract.cfm?uri=boe-5-10-3568
    Dataset : http://people.duke.edu/~sf59/Srinivasan_BOE_2014_dataset.htm
    '''
    #train_size = 2916
    #test_size = 315

    train_path = data_dir+'/Train'
    test_path = data_dir+'/Test'

    classes=['AMD', 'DME', 'NORMAL']

    train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   rescale=1.0/255,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_batches = train_datagen.flow_from_directory(train_path, target_size=(image_size,image_size),color_mode='rgb', classes=classes, batch_size=batch_size,class_mode='categorical')
    test_batches = test_datagen.flow_from_directory(test_path, target_size=(image_size,image_size),color_mode='rgb', classes=classes, batch_size=batch_size, class_mode='categorical')

    return train_batches, test_batches