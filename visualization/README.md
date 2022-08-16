 这些代码是关于把Apollo中的车道线检测的输出结果（每个车道线有几个坐标，Apollo的具体输出结果在/home/apollo/debug_out/lane下），具体的车道线检测结果输出位置可以在这块修改（在lane_detection_component.cc中可以修改）

将官方recorder中的视频保存成照片在/home/png下，修改具体代码在visulizer.cc中第1378行。

然后分别将输出的/lane文件夹和/png文件夹放到/img_and_lane文件夹下，执行程序就可以了。

具体的程序：

gui_point.py是把lane中的点画到图片上。

gui_line.py是把lane中的点拟合成线画到图片上（就是把lane中的比较少的点利用插值得到更多的点画在图片上）。

img2video.py是把上述处理完的图片合成视频。

剩下的两个py没用，是为了测试解决过拟合的问题（最后不用解决，因为这样才能真实的反应检测结果）
