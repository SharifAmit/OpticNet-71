from sklearn.metrics import confusion_matrix, classification_report
import pycm
import numpy as np

def Weighted_Error(y_true,y_pred):

    matrix = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))

    matrix[[0,3],:] = matrix[[3,0],:]
    matrix[:,[0,3]] = matrix[:,[3,0]]
    matrix[[1,3],:] = matrix[[3,1],:]
    matrix[:,[1,3]] = matrix[:,[3,1]]
    matrix[[1,2],:] = matrix[[2,1],:]
    matrix[:,[1,2]] = matrix[:,[2,1]]

    weighted_error_table = np.array([[0, 1 , 1, 1],[1,0,1,1],[4,2,0,1],[4,2,1,0]])

    weight_sum = 0
    for i in range(4):
        for j in range(4):
            if i!=j:
                weight_sum = weight_sum + (matrix[i][j]*weighted_error_table[i][j])
                
    print ('Weighted Error : '+str(weight_sum*100/1000)+'%')


def print_metric(y_true,y_pred,weighted_error=False):


    cz = pycm.ConfusionMatrix(actual_vector=y_true.argmax(axis=1), predict_vector=y_pred.argmax(axis=1))
    
    # Accuracy
    acc = cz.Overall_ACC
    print("Average Accuracy : "+str(acc*100)+'%')

    # Specificity
    specificity = cz.TNR
    totalprecision = 0
    for key, value in specificity.items():
        totalprecision = totalprecision + value
    print('Average Specificity : '+str(totalprecision*100/4.0)+'%')

    # Sensitivity
    recall = cz.TPR
    totalrecall = 0
    for key, value in recall.items():
        totalrecall = totalrecall  + value
    print('Average Sensitivity : '+str(totalrecall*100/4.0)+'%')

    if weighted_error==True:
        Weighted_Error(y_true,y_pred)

    
    

    


