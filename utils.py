#
# utils.py
# ==============================================================================
import time
import math
from contextlib import contextmanager
import operator
import re
import pickle
from functools import partial


re_sc = re.compile('[\!@#$%\^&\*\(\)-=\[\]\{\}\.,/\?~\+\'"|]')
hangul = re.compile('[^ ㄱ-ㅣ가-힣]+')
abnormal_list = ['알수 없음', '알수없음'] + \
                ['상세정보참조', '[상품상세설명 참조]', '상세페이지참조', '상세페이지 참조', 
                 '상세설명참조', '상세참조', '상품상세정보' ] + \
                ['상품상세설명'] + \
                ['추가비용', '추가 비용'] + \
                ['영업상문제로공개불가', '？？', '？？？？？？？？？', '-9'] + \
                ['별도표기', '해당없음', '알수', '없음'] + \
                "상품 상세 페이지에 제공 기타 무료 배송 참고 설명 정보 참조".strip().split()

cate_names_path = "data/final_cate_names.pickle"
f = open(cate_names_path, "rb")
cate_names = pickle.load(f)
f.close()

@contextmanager
def timer(title):
    print("Start {}".format(title))
    i_time = time.time()
    yield
    print("Finish {} - {:.0f}s".format(title, time.time()-i_time))
    
def change_special(re_sc, product):
    return re_sc.sub(" ", product).strip()
change_special = partial(change_special, re_sc)

def change_abnormal(abnormal_list, string):
    for abnormal in abnormal_list:
        string = string.replace(abnormal, " ")
    return string
change_abnormal = partial(change_abnormal, abnormal_list)
    
def get_no_hangul(s):
    return [i.lower() for i in hangul.findall(s) if len(i) > 2]

def cate_counter(sent):
    return {cate_word: sent.count(cate_word)+2 
                   for cate_word in cate_names if sent.count(cate_word) > 0}

def khaiii_api_tokenizer(khaiii_api, string):
    try:
        words = []
        for word in khaiii_api.analyze(string):
            for m in word.morphs:
                if m.tag in ["NNG", "NNP", "NNB"]:
                    words.append(m.lex)
        return words
    except:
        print("Empty sentence")
        return []

def get_int(x):
    if math.isnan(x):
        return -1
    else:
        return int(x)