import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import pickle
import numpy as np

print('importing full_data...')
full_data = pd.concat([pd.read_json(f_name, lines=True) for f_name in glob('data/json_version_chunk/*.json')])

full_data = full_data.loc[full_data['bcateid']!=-1]
cateid_data = full_data[['bcateid','mcateid','scateid','dcateid']]

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