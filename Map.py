# MaP: Mean average Precision: Mean of Average precision over all classes
#Average precision: Area under precision recall curve for a given class
#Prescion, Recall Curve: Plot of P vs R for diffrent confidence thresholds

import numpy as np
import os
import matplotlib.pyplot as plt

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

def Converter(box):
    x,y,w,h = box
    x,y,w,h = float(x), float(y), float(w), float(h)
    return ((x-w/2,y-h/2,x+w/2,y+h/2))

def p_r(pred,Gt,c,Conf_thres,IoU_thresh = 0.5):
    """
    return Precision, recalls for a given class, Conf_thres and IoU_thresh
    """
    files = os.listdir(pred)
    total_p = 0
    true_p = 0
    total_t = 0
    for f in files:
        f_p = open(os.path.join(pred,f))
        d_p = f_p.readlines() # fp is 1d
        f_p.close()
        f_t = open(os.path.join(Gt,f))
        d_t = f_t.readlines()
        f_t.close()
        f_p = [i.split() for i in d_p] # fp is 2d
        f_t = [i.split() for i in d_t]
        for box in f_p:
            if(int(box[0])==c and float(box[-1])>=Conf_thres):
                total_p+=1 # doubt
                for box_t in f_t:
                    if(box__iou(Converter(box[1:5]),Converter(box_t[1:5]))>=IoU_thresh and int(box_t[0])==c):
                        true_p+=1
                        #box_t[0] = -1
                        break
        for box in f_t:
            if(int(box[0])==c):
                total_t+=1
    return (true_p/total_p if total_p else 1,true_p/total_t)

def Ap(pred,Gt,c,IoU_thresh=0.5):
    """
    returns the Ap over the precision recall curve for a given class
    """
    Conf_thresh_limit = np.arange(0,1,0.005)  # change this to change the number of points in the curve
    P = [0]
    R = [1]
    for Conf_thresh in Conf_thresh_limit:
        p,r = p_r(pred,Gt,c,Conf_thresh,IoU_thresh)
        P.append(p)
        R.append(r)
    ans = 0
    R.append(0)
    P.append(1)
    plt.plot(R,P)
    for i in range(len(R)-1):
        ans+=(R[i]-R[i+1])*P[i]
        # print(abs(R[i+1]-R[i])*P[i])
    return ans
            

def MaP(pred,Gt,nc, IoU_thresh):

    """
    pred: Path to folder containing pred text files
    Gt: Path to folder containing Ground truth text files
    nc: No.of Classes
    IoU_thresh: Obvious
    
    returns MaP (float)
    """
    
    ans = 0
    for i in range(nc):
        ans+=Ap(pred,Gt,i,IoU_thresh)
        print(f"Ap for class {i} is {Ap(pred,Gt,i,IoU_thresh)}")
    return ans/nc


print("MaP over all the class is: ",MaP(r"C:\Users\Jignesh\Desktop\Summer_Intern\yolov5\runs\detect\exp7\labels",r"C:\Users\Jignesh\Desktop\Summer_Intern\IIIT-AR-13K_dataset\labels\test",5,0.5))
plt.show()

