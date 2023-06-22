import os
import numpy as np

def box__iou(b1,b2):
    """
    values in 4th quad
    returns IOU value of boxes (xmin,ymin,xmax,ymax) 
    """
    x1,y1,x2,y2 = b1
    x3,y3,x4,y4 = b2
    c_width = min(x2,x4)-max(x1,x3)
    c_height = min(y2,y4)-max(y1,y3)
    c_ar=c_width*c_height
    return max(0,c_ar/((x2-x1)*(y2-y1) + (x4-x3)*(y4-y3) - c_ar))

def matching(truth1,pred1,class_idx,iou_threshold):
    """
        truth: list of strings read from a single txtfile in truth
        pred: list of strings read from a single txtfile in pred
        class_idx: class for which the precision is being returned
        iou_threshold: obvious
        returns the precision for class_idx
    """
    sache_bol=0
    bol=0
    truth = []
    pred=[]
    for i in range(len(truth1)):
        truth.append(truth1[i].split())
        for j in range(len(truth[i])):
            truth[i][j]=float(truth[i][j])
    for i in range(len(pred1)):
        pred.append(pred1[i].split())
        for j in range(len(pred[i])):
            pred[i][j]=float(pred[i][j])
        if pred[i][0]==class_idx:
            px1=pred[i][1]-pred[i][3]/2
            px2=pred[i][1]+pred[i][3]/2
            py1=pred[i][2]-pred[i][4]/2
            py2=pred[i][2]+pred[i][4]/2
            bol+=1
            for i1 in range(len(truth)):
                if truth[i1][0]==class_idx:            
                    tx1=truth[i1][1]-truth[i1][3]/2
                    tx2=truth[i1][1]+truth[i1][3]/2
                    ty1=truth[i1][2]-truth[i1][4]/2
                    ty2=truth[i1][2]+truth[i1][4]/2
                    if(box__iou((px1,py1,px2,py2),(tx1,ty1,tx2,ty2))>=iou_threshold):
                        sache_bol+=1
                        break
    return (sache_bol,bol)
            
    

def xyz (truth_path, pred_path,num_classes, threshold=0.5):
    """
    truth_path: path to folder having txt files of ground truth(test set)
    pred_path: path to folder having txt files created by running detect.py (on test set)
    num_classes: obvious
    
    """
    ans = 0
    l=os.listdir(truth_path)
    l1=os.listdir(pred_path)
    for i in range(len(l1)):
        print(i)
        truth_txt=os.path.join(truth_path,l1[i])
        pred_txt=os.path.join(pred_path,l1[i])
        f=open(truth_txt,'r')
        f1=open(pred_txt,'r')
        true_data=f.readlines()
        pred_data=f1.readlines()
        # print(true_data)
        sum=0
        n_classes=0
        for j in range(num_classes):
            a,b=matching(true_data,pred_data,j,threshold)
            if b!=0:
                sum+=a/b
                n_classes+=1

        ans+=sum/n_classes
    return ans/len(l)
        
            
print(xyz(r"C:\Users\Lenovo\Desktop\IIIT-13K\labels\test",r"C:\Users\Lenovo\yolov5\runs\detect\exp4\labels",5))

# matching(["1 2 3 4 5","0 4 5 6 7" ],["1 2 3 4 5","0 4 5 6 7" ],1)
    
