import cv2
import pandas as pd

img = cv2.imread('irabu_zhang1.bmp') 

a = []
b = []
R = []
G = []
B = []

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        a.append(x)
        b.append(y)
        R.append(img[y,x,0])
        G.append(img[y,x,1])
        B.append(img[y,x,2])
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 0, 0), thickness=1)
        cv2.imshow("image",img)



cv2.namedWindow("image",cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)
cv2.waitKey(0)

data = [R,G,B]
data = pd.DataFrame(data, index=['R', 'G', 'B'])
data = data.T
data.to_csv('pix.csv')