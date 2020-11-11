import os

test_data_dir = "/home/jiamengzhao/repos/video_related/pytorch-i3d/tools/flow_in_2/1021753701_1583046019759_6.057684351283355_12.437557196131342.avi"

pics =os.listdir(test_data_dir)

pics.sort()


print(pics)

for i in pics:
    sort_i = int(i.replace('.jpg',''))
    if sort_i/2==0:
        new_i = str(sort_i)+'y.jpg'
    else:
        new_i = str(sort_i)+'x.jpg'
    print(new_i)
    cmd = 'mv '+os.path.join(test_data_dir,i)+' '+os.path.join(test_data_dir,new_i)
    print(cmd)
    os.system(cmd)
    # exit()

