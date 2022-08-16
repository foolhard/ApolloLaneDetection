import cv2
import os

#this function is intended to load lines
def load_lines(path_dir='/home/fu/img_and_lane/lane'):
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

#tihs func is intended to draw points in *.jpg
def draw_points(points,image):
    for i, j in points:
        image = cv2.circle(image, (int(i), int(j)),5, (255,0,0), -1)
    return image


#this func is intended to wirte points in *.jpg
def img_show(_path_dir):
    _points_all = load_lines(_path_dir) 
    for i in range(len(_points_all)):
        img_path  = _path_dir+'/../img/'+'{:.6f}.png'.format(i)
        img= cv2.imread(img_path)
        img_path_w = _path_dir+'/../img_w/'+'{}.jpg'.format(i)
        cv2.imwrite(img_path_w, draw_points(_points_all[i],img))

if __name__ == '__main__':
    __path_dir = '/home/fu/img_and_lane/lane'
    img_show( __path_dir)          
