# 
# gen_post_tools.py
# ==============================================================================
import math
import pickle

from glob import glob
import pandas as pd

from utils import get_int
import warnings
warnings.filterwarnings("ignore")

train_pred_n_path = "prediction_n/predict.train.top_n.tsv"
val_pred_n_path = "prediction_n/predict.val.top_n.tsv"

val_n_pdf = pd.read_csv(val_pred_n_path, sep='\t', header=None)
val_pdf = val_n_pdf.iloc[:,[0,2,3,4,5]]
val_pdf.rename(columns={0: 'pid', 
                        2: 'b_pre', 
                        3: 'm_pre', 
                        4: 's_pre', 
                        5: 'd_pre'}, 
               inplace=True)
val_pdf['d_pre'] = val_pdf['d_pre'].apply(get_int)
print(val_pdf.shape)

train_n_pdf = pd.read_csv(train_pred_n_path, sep='\t', header=None)
train_pdf = train_n_pdf.iloc[:,[0,2,3,4,5]]
train_pdf.rename(columns={0: 'pid', 
                          2: 'b_pre', 
                          3: 'm_pre', 
                          4: 's_pre', 
                          5: 'd_pre'}, 
                 inplace=True)
train_pdf['d_pre'] = train_pdf['d_pre'].apply(get_int)
print(train_pdf.shape)

data_pred = pd.concat([train_pdf, val_pdf])
data = pd.concat([pd.read_json(f_name, lines=True) 
                  for f_name in glob('data/json_version_chunk/*.json')])
_data = data[["pid", "bcateid", "mcateid", "scateid", "dcateid"]]
_data = data_pred.merge(_data, on="pid", how='inner')

_data_1 = _data[(_data.s_pre != -1) & (_data.d_pre == -1)]
_data_2 = _data[(_data.s_pre == -1) & (_data.d_pre == -1)]

_data_2_prob = _data_2[_data_2.scateid != -1]
b = _data_2_prob.groupby(['b_pre', 'm_pre'])['scateid'] \
                .apply(lambda x: x.value_counts().head(1))
_data_1_prob = _data_1[_data_1.dcateid != -1]
d = _data_1_prob.groupby(['b_pre', 'm_pre', ])['dcateid'] \
                .apply(lambda x: x.value_counts().head(1))

_data_r1 = _data[(_data.scateid != -1)]
_r_scateid = _data_r1.groupby(['b_pre'])['scateid'] \
                     .apply(lambda x: x.value_counts().head(1))
_data_r2 = _data[(_data.dcateid != -1)]
_r_dcateid = _data_r2.groupby(['b_pre'])['dcateid'] \
                     .apply(lambda x: x.value_counts().head(1))

with open("post_processing_model/tools_map.pkl", "wb") as f:
    pickle.dump([b, d, _r_scateid, _r_dcateid], f)


data = pd.concat([pd.read_json(f_name, lines=True) 
                  for f_name in glob('data/json_version_chunk/*.json')])
data = data.loc[data['bcateid']!=-1]
cateid_data = data[['bcateid','mcateid','scateid','dcateid']]

print('making pds pickle...')
# P(d_cateid|b,m,s,d_cateid==-1)
pds_data = cateid_data.loc[cateid_data['scateid']!=-1, ['bcateid','mcateid','scateid','dcateid']]
pds_data['count'] = 0
pds_data = pds_data.groupby(['bcateid','mcateid','scateid','dcateid'], as_index=False).count()
pds_data_2 = pds_data.loc[pds_data['dcateid']!=-1]
pds_data_2['count_sum'] = pds_data_2.groupby(['bcateid','mcateid','scateid'])['count'].transform(sum)
pds_data_2['prob'] = pds_data_2['count'] / pds_data_2['count_sum']

idx = pds_data_2.groupby(['bcateid','mcateid','scateid'])['prob'].transform(max) == pds_data_2['prob']
fin_pds_data = pds_data_2[idx][['bcateid','mcateid','scateid','dcateid']]

with open("post_processing_model/ml_pds_data.pickle", 'wb') as _f:
    pickle.dump(fin_pds_data, _f)

print('makiing psm pickle...')
# P(s_cateid|b,m,s_cateid==-1)
psm_data = cateid_data.loc[cateid_data['mcateid']!=-1, ['bcateid','mcateid','scateid']]
psm_data['count'] = 0
psm_data = psm_data.groupby(['bcateid','mcateid','scateid'], as_index=False).count()
psm_data_2 = psm_data.loc[psm_data['scateid']!=-1]
psm_data_2['count_sum'] = psm_data_2.groupby(['bcateid','mcateid'])['count'].transform(sum)
psm_data_2['prob'] = psm_data_2['count'] / psm_data_2['count_sum']

idx = psm_data_2.groupby(['bcateid','mcateid'])['prob'].transform(max) == psm_data_2['prob']
fin_psm_data = psm_data_2[idx][['bcateid','mcateid','scateid']]

with open("post_processing_model/ml_psm_data.pickle", 'wb') as __f:
    pickle.dump(fin_psm_data, __f)