# AVTNet

- 数据集结构：

  - data_root

        - neg1_audio
        - neg2_audio
        - pos_audio
        - neg1_audio_npy
        - neg2_audio_npy
        - pos_audio_npy
        
        - neg1_video
        - neg2_video
        - pos_videoå
        - video_npy
        
        - pos_text_npy
        - neg2_text_npy
        
        - pics_dir
        - pics_dir_cut
        
        - label_train.json
        - label_test.json
        
        - pos_text.txt
        - neg2_text.txt
        
        - text_m_all.npy

- 一键脚本(训练): /Path_to_AudioVideoNet/repos/main_entrance.py

  - 集成下列所有脚本

    - 用法：

      ```
      python3 main_entrance.py --data_root "形如上述结构的data root" 
                               --project_root "/Path_to_AudioVideoNet"
                               --select_num "num for topic matrix"
      ```

- 一键脚本(infer): /Path_to_AudioVideoNet/repos/main_entrance.py

  - 集成下列所有脚本

    - 用法：

      ```
      python3 main_entrance.py --data_root "形如上述结构的data root" 
                               --project_root "/Path_to_AudioVideoNet"
                               --select_num "num for topic matrix"
      ```


- Audio (.wav)

  - 提取特征：/Path_to_AudioVideoNet/repos/audio_related/python_speech_features

    - 用法：

      ```
      python3 example.py -i "wav files dir" 
                         -o "npy files dir"
      ```

      

      - Example: 

        ```
        python3 example.py -i ./audio
                           -o ./audio_npy
        ```

        

    - 输出：.npy文件, 保存Audio特征

    - 注意：

      - 特征提取后的维度为（t_a，13） 其中 t_a与音频时长正相关。



- Video（.avi）

  - 处理流程：video->pics->face->padding faces->npy

  - 1. 转化为图片：/Path_to_AudioVideoNet/repos/preprocess_face/prepro4video/video2pics.py

    - 用法：

      ```
      python3 video2pics.py -i "avi files dir" -o "out pics dirs root"
      ```

      

      - Example: 

        ```
        python3 video2pics.py -i ./video -o ./video_pic
        ```

        

    - 输出：在-o指定的文件夹里生成与-i文件夹中**视频名称**相同的文件夹，并把切分好的图片存入文件夹内

  - 2. 识别人脸：/Path_to_AudioVideoNet/repos/preprocess_face/Pytorch_Retinaface

    - 用法：

      ```
      python3 detect.py -i 'input pics dirs root' -m 'model path'
      ```

      		Example:

      - ```
        python3 detect.py -i ./video_pic -m /Path_to_AudioVideoNet/repos/preprocess_face/Pytorch_Retinaface/Retinaface_model_v2/Resnet50_Final.pth
        ```

        

    - 输出：在-i输入的文件夹同级的带_cut后缀文件夹中生成切割好人脸的图像对应存入以视频名称命名的子文件夹中

  - 3. padding人脸： /Path_to_AudioVideoNet/repos/video_related/pytorch-i3d/tools/padding_faces.py

    - 用法：

      ```
      python3 padding_faces.py -i ’pics_dir_cut‘
      ```

  - 输出：原地padding被切割好的人脸图片

- 4. 提取特征：/Path_to_AudioVideoNet/repos/video_related/pytorch-i3d

  - 用法：

    - 1. 构建json：/Path_to_AudioVideoNet/repos/video_related/pytorch-i3d/tools/

      - 用法：

        ```
          python generate_fake_json.py -i face_root 
                                       -o ./face.json
        ```

        

        - Example: 

          ```
            python generate_fake_json.py -i ./video_pic_cut 
                                         -o ./face.json
          ```

          

      - 输出：-o指定路径的输出json文件

    - 2. 提取特征：/Path_to_AudioVideoNet/repos/video_related/pytorch-i3d/extract_features.py

      - 用法: 

        ```
          python extract_features.py -root ./path_to_face 
                                     -split ./face.json 
                                     -save_dir ./path_to_video_npy
                                     -load_model /Path_to_AudioVideoNet/repos/video_related/pytorch-i3d/models/rgb_charades.pt
        ```

        

        - Example:

          ```
            python extract_features.py -root ./video_pic_cut 
                                       -split ./face.json 
                                       -save_dir ./video_npy
                                       -load_model /Path_to_AudioVideoNet/repos/video_related/pytorch-i3d/models/rgb_charades.pt
          ```

          

      - 注意：

        - extract_[features.py](http://features.py/) 脚本75行 if t > 100，其中100可根据cuda memory调整大小 比如 800、1600……

        - 特征提取后的维度为（t_v，1，1，1024） 其中 t_v与视频时长正相关。

          

- Text(.txt)

  - Text文本注意格式：共分两列，第一列为音频名，第二列为文本信息。

  - 词嵌入：/Path_to_AudioVideoNet/repos/AudioVideoNet/tools/

    - 用法：

      ```
      python bert_try.py -i text.txt -o text_npy_dir
      ```

    输出：存储文本相关npy的文件夹。

- 后处理（.npy）


  - 1.padding：将npy搞成一致的长度 （只处理音频和视频，文本自动padding）

    - 脚本：/home/jiamengzhao/repos/AudioVideoNet/tools/

      - 用法：

        ```
        python padding_npys.py -i data_root
        ```

      - 注意：data_root为一个根目录文件夹，包含所有的预处理npy文件夹。结构应该如下所示：

        - data_root

          - neg1_audio
          - neg2_audio
          - pos_audio
          - neg1_audio_npy
          - neg2_audio_npy
          - pos_audio_npy

          - neg1_video
          - neg2_video
          - pos_video
          - video_npy

          - pos_text_npy
          - neg2_text_npy

          - pics_dir
          - pics_dir_cut



- 2. 生成训练所用json: 记录文件位置，分割数据集为train和test

  - 脚本：/home/jiamengzhao/repos/AudioVideoNet/tools/

    - 用法：

      ```
      python generate_json.py --data_root data_root --train_label_json train.json  --test_label_json test.json
      ```

  3. 生成用于提炼topics的原始矩阵：随机抽一部分positive sample的text npy 生成一个大一点的矩阵 存成npy

  - 脚本：/home/jiamengzhao/repos/AudioVideoNet/tools

    - 用法：

      ```
      python get_topics_dic.py --train_label /path_to_json/label_json.json
      												 --out_m_npy   /path_to_topic_matrix
      												 --data_root   /data_root
                               --select_num num_for_topic_matrix
      ```

      Example:

      ```
      python get_topics_dic.py --train_label ./data_root/label_json.json
      												 --out_m_npy   ./data_root/topic.npy
      												 --data_root   ./data_root
                               --select_num 300
      ```

      
