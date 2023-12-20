import os
import re
import shutil
origin_path = r'/home/vtie/Desktop/txcjycl/alexnet_dataset'
target_path_0 = r'/home/vtie/Desktop/txcjycl/alexnet_train/0'
target_path_1 = r'/home/vtie/Desktop/txcjycl/alexnet_train/1'
 
file_list=os.listdir(origin_path)
print(len(file_list))

for i in range(len(file_list)):
    old_path=os.path.join(origin_path,file_list[i])
    print(file_list[i])
    result=re.findall(r'\w+',file_list[i])[0]

    if result=='cat':
        shutil.move(old_path,target_path_0)
    else:
      
        shutil.move(old_path,target_path_1)