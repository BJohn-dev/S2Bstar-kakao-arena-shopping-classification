# S2B-kakao-arena-shopping-classification



## :whale: 시스템 정보
> nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04  
python3.6.4 | python3.6.7


## :whale: 추가 라이브러리
> [khaiii](https://github.com/kakao/khaiii)


## :whale: 모델 생성
#### 0. 데이터의 위치
>프로젝트 상위에 위치  
####  1. chunck 파일을 json 형태로 저장
> $ python3 save_json_version_chunk.py train   
$ python3 save_json_version_chunk.py dev  
$ python3 save_json_version_chunk.py test

####  2. Embedding 네트워크 학습을 위한 데이터 만들기
> $ python3 gen_cate_keyword.py
$ python3 data.py build_y_vocab
$ python3 data.py make_db train

####  3. 임베딩 네트워크 학습
> $ python3 embedding_classifier.py train data/train ./embedding_model/train

####  4. 임베딩 네트워크를 이용해 단어 임베딩 특징 추가
> $ python3 embedding_apply.py train

####  5. 예측 모델 학습
> $ python3 classifier.py train data/train model/train 

####  6. 후처리 모델 준비
> `$ python3 -u classifier_top_n_prediction_train_val.py predict data/train ./model/train data/train/ train prediction_n/predict.train.top_n.tsv`   
`$ python3 -u classifier_top_n_prediction_train_val.py predict data/train ./model/train data/train/ dev prediction_n/predict.val.top_n.tsv`    
$ python3 make_map_pkl.py  
$ python3 make_ml_pkl.py  


## :whale: dev 데이터 예측하기
> $ python3 data.py make_db dev ./data/dev --train_ratio=0.0 (ccc)
$ python3 embedding_apply.py dev (ccc)
$ python3 inference.py dev (ccc-b)


## :whale: test 데이터 예측 진행하기
> $ python3 data.py make_db test ./data/test --train_ratio=0.0   
$ python3 embedding_apply.py test    
$ python3 inference.py test    


## :whale: 미리 생성된 모델을 바탕으로 test 데이터 예측 진행하기
#### 0. 미리 생성된 모델 리소스 다운
> $ wget https://www.dropbox.com/s/z8cr91uey32fyta/tools_map.pkl?dl=0 -O post_processing_model/tools_map.pkl  
$ wget https://www.dropbox.com/s/kuhxfkn2rsfkj8l/ml_psm_data.pickle?dl=0 -O post_processing_model/ml_psm_data.pickle  
$ wget https://www.dropbox.com/s/np82vh1dp0hedfj/ml_pds_data.pickle?dl=0 -O post_processing_model/ml_pds_data.pickle  
$ mkdir model/train  
$ wget https://www.dropbox.com/s/54p2rh83tvudqw0/weights?dl=0 -O model/train/weights  

#### 1. 추론 시작
> $ python3 inference.py test 


## :whale: 내용 추가 할 것  
> - Test 파일과 Dev 파일 업로드 하기 (드롭박스) ***    
> - 사용된 데이터셋 구글 드라이브에 저장하기       
> - NOTICE 라이센스  
> - 개별 파일 상단 라이센스 확인  
> - Public 으로 바꾸기 ***  