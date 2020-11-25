import os
import json

lmks_dir = "/home/jiamengzhao/data_root/new_test_data_root/lmks"
rm_data = []
for i in os.listdir(lmks_dir):
    ids =[]
    count = 0
    for j in os.listdir(os.path.join(lmks_dir,i)):
        if '.npy' in j:
            ids.append(int(j.replace('.npy','')))
            count= count+1
    list.sort(ids)

    print(ids[-1],count)
    if ids[-1] !=count:
        rm_data.append(i)

json_path = "/home/jiamengzhao/data_root/new_test_data_root/train_label_pos.json"
with open(json_path) as file:
    dic = json.load(file)

print(dic.keys())

for single_data in rm_data:
    del dic[single_data]


def write_json_file(json_path, json_dic):
    with open(json_path, "w") as f:
        json.dump(json_dic, f, indent=4, ensure_ascii=False)
    
write_json_file(json_path,dic)


