import math
import numpy as np
import pandas as pd
import cv2



def coVariance(X):
    ro, cl = X.shape
    row_mean = np.mean(X,axis=0)
    X_Mean = np.zeros_like(X)
    X_Mean[:] = row_mean
    X_Minus = X - X_Mean
    covarMatrix = np.zeros((cl,cl))
    for i in range(cl):
        for j in range(cl):
            covarMatrix[i,j] = (X_Minus[:,i].dot(X_Minus[:,j].T)) / (ro-1)
    return covarMatrix



def getLikelihood(x,mean,std,x_inv):
    std = math.pow(std,1/2)
    exponent = np.exp(-(x-mean)@(2*x_inv)@(x-mean).T)
    print(exponent)
    return (1/(np.sqrt(2*math.pi)*std))*exponent

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        X = [img[y,x,0],img[y,x,1],img[y,x,2]]

        prob = np.zeros(5)
        for i in range(5):
            prob[i] = getLikelihood(X,m[i],x_norm[i],x_inv[i])
            label = str(np.argmax(prob)+1)
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 0, 0), thickness=1)
        cv2.imshow("image",img)




datafra = pd.read_csv("./pix_tmp.csv")

field1 = np.array(datafra[["R","G","B"]][:10])
field2 = np.array(datafra[["R","G","B"]][:20])
field3 = np.array(datafra[["R","G","B"]][:30])
field4 = np.array(datafra[["R","G","B"]][:40])
field5 = np.array(datafra[["R","G","B"]][:50])

m = np.zeros([5,3])
m[0] = [np.mean(field1[:,0]),np.mean(field1[:,1]),np.mean(field1[:,2])]
m[1] = [np.mean(field2[:,0]),np.mean(field2[:,1]),np.mean(field2[:,2])]
m[2] = [np.mean(field3[:,0]),np.mean(field3[:,1]),np.mean(field3[:,2])]
m[3] = [np.mean(field4[:,0]),np.mean(field4[:,1]),np.mean(field4[:,2])]
m[4] = [np.mean(field5[:,0]),np.mean(field5[:,1]),np.mean(field5[:,2])]

covarMatrix1 = coVariance(field1)
covarMatrix2 = coVariance(field2)
covarMatrix3 = coVariance(field3)
covarMatrix4 = coVariance(field4)
covarMatrix5 = coVariance(field5)

x_norm = np.zeros(5)
x_norm[0]=np.linalg.norm(covarMatrix1, ord=None, axis=None, keepdims=False)
x_norm[1]=np.linalg.norm(covarMatrix2, ord=None, axis=None, keepdims=False)
x_norm[2]=np.linalg.norm(covarMatrix3, ord=None, axis=None, keepdims=False)
x_norm[3]=np.linalg.norm(covarMatrix4, ord=None, axis=None, keepdims=False)
x_norm[4]=np.linalg.norm(covarMatrix5, ord=None, axis=None, keepdims=False)


x_inv = np.zeros([5,3,3])
x_inv[0] = np.linalg.inv(covarMatrix1)
x_inv[1] = np.linalg.inv(covarMatrix2)
x_inv[2] = np.linalg.inv(covarMatrix3)
x_inv[3] = np.linalg.inv(covarMatrix4)
x_inv[4] = np.linalg.inv(covarMatrix5)


img = cv2.imread('./irabu_zhang1.bmp') 
cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)
cv2.waitKey(0)
