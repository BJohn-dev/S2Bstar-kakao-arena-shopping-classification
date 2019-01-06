import math
import time
from contextlib import contextmanager
import pickle

import pandas as pd

from glob import glob


@contextmanager
def timer(title):
    print("Start {}".format(title))
    i_time = time.time()
    yield
    print("Finish {} - {} s".format(title, time.time()-i_time))

def get_int(x):
    if math.isnan(x):
        return -1
    else:
        return int(x)

train_pred_n_path = "prediction_n/predict.train.top_n.tsv"
val_pred_n_path = "prediction_n/predict.val.top_n.tsv"

val_n_pdf = pd.read_csv(val_pred_n_path, sep='\t', header=None)
val_pdf = val_n_pdf.iloc[:,[0, 2,3,4,5]]
val_pdf.rename(columns={0: 'pid', 2: 'b_pre', 3: 'm_pre', 4: 's_pre', 5: 'd_pre'}, inplace=True)
val_pdf['d_pre'] = val_pdf['d_pre'].apply(get_int)
print(val_pdf.shape)

train_n_pdf = pd.read_csv(train_pred_n_path, sep='\t', header=None)
train_pdf = train_n_pdf.iloc[:,[0, 2,3,4,5]]
train_pdf.rename(columns={0: 'pid', 2: 'b_pre', 3: 'm_pre', 4: 's_pre', 5: 'd_pre'}, inplace=True)
train_pdf['d_pre'] = train_pdf['d_pre'].apply(get_int)
print(train_pdf.shape)

data_pred = pd.concat([train_pdf, val_pdf])

data = pd.concat([pd.read_json(f_name, lines=True) for f_name in glob('data/json_version_chunk/*.json')])
data = data[["pid", "bcateid", "mcateid", "scateid", "dcateid"]]
_data = data_pred.merge(data, on="pid", how='inner')


_data_1 = _data[(_data.s_pre != -1) & (_data.d_pre == -1)]
_data_2 = _data[(_data.s_pre == -1) & (_data.d_pre == -1)]
# _data_3 = _data[(_data.s_pre == -1) & (_data.d_pre != -1)]
# _data_4 = _data[(_data.s_pre != -1) & (_data.d_pre != -1)]


_data_2_prob = _data_2[_data_2.scateid != -1]
b = _data_2_prob.groupby(['b_pre', 'm_pre'])['scateid'].apply(lambda x: x.value_counts().head(1))
_data_1_prob = _data_1[_data_1.dcateid != -1]
d = _data_1_prob.groupby(['b_pre', 'm_pre', ])['dcateid'].apply(lambda x: x.value_counts().head(1))


_data_r1 = _data[(_data.scateid != -1)]
_r_scateid = _data_r1.groupby(['b_pre'])['scateid'].apply(lambda x: x.value_counts().head(1))

_data_r2 = _data[(_data.dcateid != -1)]
_r_dcateid = _data_r2.groupby(['b_pre'])['dcateid'].apply(lambda x: x.value_counts().head(1))


f = open("post_processing_model/tools_map.pkl", "wb")
pickle.dump([b, d, _r_scateid, _r_dcateid], f)
f.close()

# def get_new_s_pre(b_pre, m_pre, s_pre):
#     if s_pre == -1:
#         try:
#             return b.loc[(b_pre, m_pre)].index.values[0]
#         except:
#             return s_pre
        
#     return s_pre
        
# def get_new_d_pre(b_pre, m_pre, d_pre):
#     if d_pre == -1:
#         try:
#             return d.loc[(b_pre, m_pre)].index.values[0]
#         except:
#             return d_pre
        
#     return d_pre


# def get_new_s_pre_only_b_pre(b_pre, s_pre):
#     if s_pre == -1:
#         try:
#             return _r_scateid.loc[b_pre].index.values[0]
#         except:
#             return s_pre
        
#     return s_pre
        
# def get_new_d_pre_only_b_pre(b_pre, d_pre):
#     if d_pre == -1:
#         try:
#             return _r_dcateid.loc[b_pre].index.values[0]
#         except:
#             return d_pre
        
#     return d_pre











