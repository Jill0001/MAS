
#{"cutpics": {"subset": "training", "duration": 2, "actions": []}}
import json
import os
from decimal import Decimal
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i',"--input_dir",help="cut pics dirs root")
parser.add_argument('-o',"--output_json",help="output json file path")
args = parser.parse_args()

# main_json_path ="/home/jiamengzhao/repos/video_related/pytorch-i3d/tools/all_json.json"
main_json_path = args.output_json
main_json = {}
# pics_root = "/home/jiamengzhao/repos/preprocess_face/prepro4video/out_pics_cut"
pics_root = args.input_dir
pics_dirs = os.listdir(pics_root)
for pics_dir in pics_dirs:
    pic_dir_path = os.path.join(pics_root,pics_dir)
    pics_id = pics_dir
    main_json[pics_id] = {}
    main_json[pics_id]["subset"] = "training"
    frames = len(os.listdir(pic_dir_path))
    duration = frames/30.0
    main_json[pics_id]["duration"] =float(format(duration, '.2f'))
    main_json[pics_id]["actions"] = []

# json_str = json.dumps(main_json)
with open(main_json_path,"w") as f:
    json.dump(main_json,f,indent=4,ensure_ascii=False)