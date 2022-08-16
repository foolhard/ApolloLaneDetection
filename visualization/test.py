import cv2
import numpy as np

def base_fucs():
    img1 = cv2.imread('project_pic/1.jpg')

    pts = np.array([[300, 100], [320, 300], [500, 310], [600, 450], [650, 600], [700, 680]])  # 随便取几个散点
    pts_fit2 = np.polyfit(pts[:, 0], pts[:, 1], 2)  # 拟合为二次曲线
    pts_fit3 = np.polyfit(pts[:, 0], pts[:, 1], 3)  # 拟合为三次曲线
    print(pts_fit2)  # 打印系数列表，含三个系数
    print(pts_fit3)  # 打印系数列表，含四个系数
    
    plotx = np.linspace(300, 699, 400)  # 按步长为1，设置点的x坐标
    ploty2 = pts_fit2[0]*plotx**2 + pts_fit2[1]*plotx + pts_fit2[2]  # 得到二次曲线对应的y坐标
    ploty3 = pts_fit3[0]*plotx**3 + pts_fit3[1]*plotx**2 + pts_fit3[2]*plotx + pts_fit3[3]  # 得到三次曲线对应的y坐标

    pts_fited2 = np.array([np.transpose(np.vstack([plotx, ploty2]))])  # 得到二次曲线对应的点集
    pts_fited3 = np.array([np.transpose(np.vstack([plotx, ploty3]))])  # 得到三次曲线对应的点集
    
    cv2.polylines(img1, [pts], False, (0, 0, 0), 5)  # 原始少量散点构成的折线图
    cv2.polylines(img1, np.int_([pts_fited2]), False, (0, 255, 0), 5)  # 绿色 二次曲线上的散点构成的折线图，近似为曲线
    cv2.polylines(img1, np.int_([pts_fited3]), False, (0, 0, 255), 5)  # 红色 三次曲线上的散点构成的折线图，近似为曲线
    cv2.namedWindow('img1', 0)
    cv2.imshow('img1', img1)

    cv2.waitKey(0)