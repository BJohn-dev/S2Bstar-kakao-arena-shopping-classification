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


## Leaderboards
**팀명: S2Bstar** (잠정 순위 3위, TEST Score 기준)  

**TEST Score** (2019.01.12)
![Test Score](/img/final_board_rank.png)

**DEV Score** (2019.01.12)
![Dev Score](/img/public_board_rank.png)
