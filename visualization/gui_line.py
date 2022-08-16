import cv2
import os
from numpy import append
import scipy.interpolate as spi
import math
import numpy as np

name = {'pos_type','image_point_set'}
#this function is intended to load lines
def load_lines(path_dir='/home/fu/img_and_lane/lane'):
    dirlist = sorted(os.listdir(path_dir))
    points_all = []
    #这是所有的点
    for i in range(len(dirlist)):
        file_path = path_dir +'/{}.txt'.format(i)
        with open(file_path,'r') as f:
            lines = eval(f.read())
            pts=[]
            for line in lines:

                x=[]
                y=[]

                line['image_point_set'] = sorted(line['image_point_set'], key= lambda i:i['y'])

                for point in line['image_point_set']:
                    if point['y'] != (y[-1] if len(y) > 0 else -10000) and point['x'] != (x[-1] if len(x) > 0 else -10000) :
                        x.append(point['x'])
                        y.append(point['y'])
                pts.append([{'x':x,'y':y}])
            points_all.append(pts)
    # print(points_all)
    return points_all
    
         
def points2manypoints():
    points_all = load_lines(path_dir='/home/fu/img_and_lane/lane')
    txt_all = []
    for points_one in points_all:
        pts_list = []
        # print(points_one)
        for pts in points_one:
            # print(pts)
            for xy in pts:
                # print(line)
                # for xy in line :
                line_one = []
                Ymax = 1080

                y = xy['y']
                x = xy['x']

                start_y = min(math.floor(max(y)),Ymax -1)
                end_y = max (math.floor(min(y)),0)

                new_y = [k for k in range(end_y,start_y+1)]
                if len(x) > 3:
                    ipo3 = spi.splrep(y,x,k=3)
                    new_x = spi.splev(new_y,ipo3)


                else :
                    ipo3 = spi.splrep(y,x,k=1)
                    new_x = spi.splev(new_y,ipo3)
                
            
                line_one.append({'x':new_x})
                line_one.append({'y':new_y})
            pts_list.append(line_one) 
        txt_all.append(pts_list)
    # print(txt_all)
    return txt_all




def draw_points(_path_dir):
    _txt_all = points2manypoints()
    # for m in range(len(_txt_all)):
      
    for m, _txt in enumerate(_txt_all):
        img_path  = _path_dir+'/../img/'+'{:.6f}.png'.format(m)
        img= cv2.imread(img_path)
        img_path_w = _path_dir+'/../img_w/'+'{}.jpg'.format(m)
        for lines in _txt:
            x = list(lines[0]['x'])
            # print(x)
            y = lines[1]['y']
            # print('sasadssasdas:::::::::::::::',y)
            xy_zip = zip(x,y)
            for i,j in xy_zip:
                img = cv2.circle(img, (int(i), int(j)),5, (255,0,0), -1)
        cv2.imwrite(img_path_w, img)


# this func is intended to wirte points in *.jpg
# def img_show(_path_dir):
#     _points_all = load_lines(_path_dir) 
#     for i in range(len(_points_all)):
#         img_path  = _path_dir+'/../img/'+'{:.6f}.png'.format(i)
#         img= cv2.imread(img_path)
#         img_path_w = _path_dir+'/../img_w/'+'{}.jpg'.format(i)
#         cv2.imwrite(img_path_w, draw_points(_points_all[i],img))

if __name__ == '__main__':
    __path_dir = '/home/fu/img_and_lane/lane'
    draw_points(__path_dir)  

