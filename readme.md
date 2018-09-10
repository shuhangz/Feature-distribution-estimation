#SIFT特征提取、过滤
svm_sift.py
##输入
1. 图像文件（tif, png, jpg）
2. 坐标文件（tfw, pgw, jgw）与图形文件名字相同，必须投影到平面
##执行
svm_sift.py
filename设置成影像文件名称
##输出
特征点结果：/svm_data/keypoints_影像文件名.csv

文件组织：x,y,size,angle,response,octave,point_id,is_keypoint

##ArcGIS
Add XY data -> csv文件
添加'x''y' 字段，导出dbf格式，坐标系WGS1984。给uav_cpp_python作为输入

