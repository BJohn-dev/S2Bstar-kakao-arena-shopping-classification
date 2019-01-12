# 
# utils_post.py
# ==============================================================================
import operator
import pickle
from functools import partial


def dict_max(dictionary):
    _a = max(dictionary.items(), key=operator.itemgetter(1))[0]
    if _a == -1:
        try: return sorted(dictionary.items(), key=lambda kv: kv[1])[-2][0]
        except: return _a
    return _a


def post_process_first_stage(line):

    cate_dict = {}
    for i in range(1,5):
        cate_dict[i] = {}
        for rank in range(5):  
                try: cate_dict[i][line[5*rank+i+1]] += line[5*rank+1]
                except: cate_dict[i][line[5*rank+i+1]] = line[5*rank+1]

    new_line = [line[0], 
                dict_max(cate_dict[1]), 
                dict_max(cate_dict[2]), 
                dict_max(cate_dict[3]), 
                dict_max(cate_dict[4])]

    for i in range(1, 5):
        for j in range(2, 6):
            new_line.append(line[i*5+j])

    return new_line


pds_df = pickle.load(open('post_processing_model/ml_pds_data.pickle','rb'))
psm_df = pickle.load(open('post_processing_model/ml_psm_data.pickle','rb'))

pds_list = pds_df[['bcateid', 'mcateid', 'scateid']].values.tolist()

def ml_changer(line, max_rank=5):
    for rank in range(max_rank):
        if line[3+rank*4] == -1.0:
            if line[1+rank*4:3+rank*4] in psm_df[['bcateid', 'mcateid']].values.tolist():
                line[3+rank*4] = psm_df.loc[(psm_df['bcateid']==line[1+rank*4])&(psm_df['mcateid']==line[2+rank*4])]['scateid'].tolist()[0]
        if line[3+rank*4] != -1.0 and line[4+rank*4] == -1.0:
            if line[1+rank*4:4+rank*4] in pds_df[['bcateid', 'mcateid', 'scateid']].values.tolist():
                line[4+rank*4] = pds_df.loc[(pds_df['bcateid']==line[1+rank*4])&(pds_df['mcateid']==line[2+rank*4])&(pds_df['scateid']==line[3+rank*4])]['dcateid'].tolist()[0]
    return line

def rank_puller(line, max_rank=3):
    if line[3]==-1:
        for rank in range(1,max_rank):
            if line[3+rank*4] != -1:
                line[3] = line[3+rank*4]
                break
    if line[4]==-1:
        for rank in range(1,max_rank):
            if line[4+rank*4] != -1:
                line[4] = line[4+rank*4]
                break
    return line[0:5]

b, d, _r_scateid, _r_dcateid = pickle.load(open('post_processing_model/tools_map.pkl', 'rb'))
def get_new_s_pre(b_pre, m_pre, s_pre):
    if s_pre == -1:
        try:
            return b.loc[(b_pre, m_pre)].index.values[0]
        except:
            return s_pre
        
    return s_pre
        
def get_new_d_pre(b_pre, m_pre, d_pre):
    if d_pre == -1:
        try:
            return d.loc[(b_pre, m_pre)].index.values[0]
        except:
            return d_pre
        
    return d_pre

def get_new_s_pre_only_b_pre(b_pre, s_pre):
    if s_pre == -1:
        try:
            return _r_scateid.loc[b_pre].index.values[0]
        except:
            return s_pre
        
    return s_pre
        
def get_new_d_pre_only_b_pre(b_pre, d_pre):
    if d_pre == -1:
        try:
            return _r_dcateid.loc[b_pre].index.values[0]
        except:
            return d_pre
        
    return d_pre
