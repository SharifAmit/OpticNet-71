import os
from shutil import copyfile

def sri2014_process():
    data_dir='2014_BOE_Srinivasan/Publication_Dataset/'
    folder_names = os.listdir(data_dir)

    base_dir = 'data/'

    os.mkdir(os.path.join(base_dir,'Srinivasan2014'))
    extended_dir= os.path.join(base_dir,'Srinivasan2014')
    list_of_set = ['Train','Test']
    list_of_dir = ['AMD','DME','NORMAL']
    for setlist in list_of_set:
        more_extended = os.path.join(extended_dir,setlist)
        os.mkdir(os.path.join(extended_dir,setlist))
        for directory in list_of_dir:
            os.mkdir(os.path.join(more_extended,directory))

    t = 0
    for x in folder_names:
        y = data_dir+ x + '/TIFFs/8bitTIFFs/'
        #print (x)
        Z = os.listdir(y)
        #print(Z)
        count = 0
        for z in Z:
            oldfile = y+z
            newfile =x+ '_'+z
            #print (oldfile)
            #print (newfile)
            if count<7: # Put into Test
                if newfile[0] == 'A':
                    copyfile(oldfile,'data/Srinivasan2014/Test/AMD/'+newfile)
                elif newfile[0] == 'D':
                    copyfile(oldfile,'data/Srinivasan2014/Test/DME/'+newfile)
                else:
                    copyfile(oldfile,'data/Srinivasan2014/Test/NORMAL/'+newfile)
            else: # Put into Train
                if newfile[0] == 'A':
                    copyfile(oldfile,'data/Srinivasan2014/Train/AMD/'+newfile)
                elif newfile[0] == 'D':
                    copyfile(oldfile,'data/Srinivasan2014/Train/DME/'+newfile)
                else:
                    copyfile(oldfile,'data/Srinivasan2014/Train/NORMAL/'+newfile)
            count = count +1
        #print(z)
        t = t + len(z)

if __name__ == '__main__':
    sri2014_process()
