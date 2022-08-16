import cv2
import os
import scipy.interpolate as spi
import math

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
    return points_all

#这个函数是把点变多   
def points2manypoints():
    points_all = load_lines(path_dir='/home/fu/img_and_lane/lane')
    txt_all = []
    for points_one in points_all:
        pts_list = []
        for pts in points_one:
            for xy in pts:

                line_one = []
                Ymax = 1080

                y = xy['y']
                x = xy['x']

                start_y = min(math.floor(max(y)),Ymax -1)
                end_y = max (math.floor(min(y)),0)

                new_y = [k for k in range(end_y,start_y+1)]
                if len(x) > 3:
                    ipo3 = spi.splrep(y,x,k=1)
                    new_x = spi.splev(new_y,ipo3)
                else :
                    ipo3 = spi.splrep(y,x,k=1)
                    new_x = spi.splev(new_y,ipo3)
            
                line_one.append({'x':new_x})
                line_one.append({'y':new_y})
            pts_list.append(line_one) 
        txt_all.append(pts_list)
    return txt_all

#这个函数是把多个点画在图上
def draw_points(_path_dir):
    _txt_all = points2manypoints() 
    for m, _txt in enumerate(_txt_all):
        img_path  = _path_dir+'/../img/'+'{:.6f}.png'.format(m)
        img= cv2.imread(img_path)
        img_path_w = _path_dir+'/../img_w/'+'{}.jpg'.format(m)
        for lines in _txt:

            x = list(lines[0]['x'])
            y = lines[1]['y']
            xy_zip = zip(x,y)
            for i,j in xy_zip:
                img = cv2.circle(img, (int(i), int(j)),5, (255,0,0), -1)
        cv2.imwrite(img_path_w, img)

#这个函数是加载原始点
def load_little_lines(path_dir='/home/fu/img_and_lane/lane'):
    dirlist = sorted(os.listdir(path_dir))
    points_all = []
    for i in range(len(dirlist)):
        file_path = path_dir +'/{}.txt'.format(i)
        with open(file_path,'r') as f:
            lines = f.read().splitlines()
            str_data = ''
            for i in lines:
                str_data =str_data + i
            data_list = eval(str_data)
            points_one = []
            for data in data_list:
                if data.get('image_point_set',None) is not None:
                    points = data['image_point_set']
                    for point in points:
                        points_one.append((point['x'],point['y']))
            points_all.append(points_one)
    return points_all

#这个函数是把原始点画在图上
def draw_point(points,image):
    for i, j in points:
        image = cv2.circle(image, (int(i), int(j)),10, (255,255,0), -1)
    return image

# 这个函数的功能是把原始点画在图上
def img_show(_path_dir):
    _points_all = load_little_lines(_path_dir) 
    for i in range(len(_points_all)):
        img_path  = _path_dir+'/../img_w/'+'{}.jpg'.format(i)
        img= cv2.imread(img_path)
        img_path_w = _path_dir+'/../img_w/'+'{}.jpg'.format(i)
        cv2.imwrite(img_path_w, draw_point(_points_all[i],img))

if __name__ == '__main__':
    __path_dir = '/home/fu/img_and_lane/lane'
    draw_points(__path_dir) 
    img_show(__path_dir)
    # points2manypoints()

