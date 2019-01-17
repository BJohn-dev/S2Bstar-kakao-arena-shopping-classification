# S2B-kakao-arena-shopping-classification

## :whale: 시스템 정보
> nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04  
python3.6.4 | python3.6.7


## :whale: 추가 라이브러리
**[khaiii](https://github.com/kakao/khaiii)**  
> khaiii_api = khaiii.KhaiiiApi('/usr/local/lib/libkhaiii.so')  
khaiii_api.open("/usr/local/share/khaiii")  
        

## :whale: 모델 생성
**데이터의 위치**  
프로젝트 상위(../)에 [제공된 데이터셋](https://arena.kakao.com/c/1/data) 다운  

**데이터셋 생성**  
> $ python3 gen_cate_keyword.py  
$ python3 data.py build_y_vocab  
$ python3 data.py make_db train  

**임베딩 네트워크 학습 및 특징 추가**
> $ python3 train_predict.py embd_train data/train ./embedding_model/train  
$ python3 apply_embd.py train

**예측 모델 학습**
> $ python3 train_predict.py train data/train model/train 

**후처리 모델 준비**
> `$ python3 -u train_predict.py predict data/train ./model/train data/train/ train prediction_n/predict.train.top_n.tsv`   
`$ python3 -u train_predict.py predict data/train ./model/train data/train/ dev prediction_n/predict.val.top_n.tsv`  
$ python3 save_json_version_chunk.py train  
$ python3 gen_post_tools.py  


## :whale: 추론 진행
> $ python3 data.py make_db test ./data/test --train_ratio=0.0   
$ python3 apply_embd.py test    
$ python3 inference.py test    


## :whale: 미리 생성된 모델을 바탕으로 test 데이터 예측 진행하기
#### 0. 미리 생성된 모델 리소스를 다운 받고, TEST 데이터셋 생성
> $ wget https://www.dropbox.com/s/z8cr91uey32fyta/tools_map.pkl?dl=0 -O post_processing_model/tools_map.pkl  
$ wget https://www.dropbox.com/s/rbfkqypwe724uq7/final_cate_names.pickle?dl=0 data/final_cate_names.pickle   
$ wget https://www.dropbox.com/s/95zl2g1244kq73x/y_vocab.py3.cPickle?dl=0 data/y_vocab.py3.cPickle   
$ mkdir embedding_model/train  
$ wget https://www.dropbox.com/s/ltpeeoga0wo5295/weights-13epoch?dl=0 -O embedding_model/train/weights   
$ wget https://www.dropbox.com/s/kuhxfkn2rsfkj8l/ml_psm_data.pickle?dl=0 -O post_processing_model/ml_psm_data.pickle  
$ wget https://www.dropbox.com/s/np82vh1dp0hedfj/ml_pds_data.pickle?dl=0 -O post_processing_model/ml_pds_data.pickle  
$ mkdir model/train  
$ wget https://www.dropbox.com/s/54p2rh83tvudqw0/weights?dl=0 -O model/train/weights  

> $ python3 data.py make_db test ./data/test --train_ratio=0.0   
$ python3 apply_embd.py test    
#### 1. test 데이터 예측 진행하기
> $ python3 inference.py test    


## :whale: Leaderboards
**팀명: S2Bstar** (잠정 순위 3위, TEST Score 기준)  

**TEST Score** (2019.01.12)
![Test Score](/img/final_board_rank.png)

**DEV Score** (2019.01.12)
![Dev Score](/img/public_board_rank.png)
