# MAS

- Dataset structure：

  - data_root

        - neg1_audio
        - neg2_audio
        - pos_audio
        
        - neg1_audio_npy
        - neg2_audio_npy
        - pos_audio_npy
        
        - lmks
       
        - text_neg_npy
        - text_pos_npy
        
        - pics_dir
        
        - label_train.json
        - label_val.json
        - label_test.json
        
        - pos_text.txt
        - neg2_text.txt
        
        - text_m_all.npy

- Data PreProcessing:
    - Audio (.wav)

        - Please use this https://github.com/jameslyons/python_speech_features to extract audio feature.

    - Video（.avi）

         - Pipeline：video->pictures->landmarks
            - video->pictures：
                - path: ./tools/video2pics.py
                - usage:
    
                  ```
                  python3 video2pics.py -i "avi files dir" -o "out pics dirs root"
                  ```
            - pictures->landmarks：
                http://dlib.net/
           
    - Text (.txt)
    
      - path ./tools/preprocess_bert.py
    
      - Usage：
    
          ```
          python preprocess_bert.py -i text.txt -o text_npy_dir
          ```
    

- Generate labels:

  - Padding：

    -  Tool path: ./tools/padding_npys.py

    - 用法：

        ```
        python padding_npys.py -i data_root
        ```
      
  - Generate label json: 

    - Tool path：./tools/generate_json.py

    - Usage：

      ```
      python generate_json.py --data_root data_root 
      ```

  - Generate positive sample matrix

    - Tool path：./tools/generate_matrix.py

    - Usage：

      ```
      python generate_matrix.py --train_label /path_to_json/label_json.json
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
      
- Train & Test：
    - Tool path:
        ./train_lmks.py
     - Usage：

      ```
      python train_lmks.py --data_root path_to_data_root
      	             --save_model_path path_to_save_models
 
      ```
        

      
