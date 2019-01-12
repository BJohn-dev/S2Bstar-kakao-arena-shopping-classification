#
# save_json_version_chunk.py
# ==============================================================================
import os
import sys
import json
import pickle as pkl
from multiprocessing import Pool

import numpy as np
import h5py

from utils import timer
from utils_class import opt

def save_chunk(div_and_path):
    
    num, div, data_path = div_and_path
    print(div_and_path)
    with timer("Parse chunk {}".format(num)):   
        train_h = h5py.File(data_path, 'r')
        train = train_h[div]

        chunk_path = os.path.join('data/json_version_chunk/', 
                                  "json_chunck.{}.{}.json".format(div,num))
        with open(chunk_path, 'w') as f:
            
            length = train['bcateid'].shape[0]
            for i in range(length):
                
                line_dict = {}
                keys = list(train.keys())
                for key in keys:
                    if key == 'img_feat':
                        continue
                    value = train[key][i]
                    if isinstance(value, np.bytes_):
                        value = value.decode()
                    elif isinstance(value, np.int32):
                        value = int(value)
                    else:
                        # Leave log!!!!!
                        break
                    
                    key = str(key)
                    line_dict[key] = value
                
                f.write(json.dumps(line_dict))
                f.write("\n")
    return


if __name__ == "__main__":

    train_data_list = opt.train_data_list
    dev_data_list = opt.dev_data_list
    test_data_list = opt.test_data_list
    
    target_div = sys.argv[1]
    target_data_list = None
    if target_div == 'dev':
        target_data_list = dev_data_list
    elif target_div == 'test':
        target_data_list = test_data_list
    elif target_div == 'train':
        target_data_list = train_data_list
        
    save_dir = 'data/json_version_chunk/' 
    print(target_div, len(target_data_list))
    if os.path.isdir(save_dir):
        pass
    else:
        os.mkdir(save_dir)

    num_workers = len(train_data_list)
    pool = Pool(num_workers)
    try:
        rets = pool.map_async(save_chunk, 
                              [(num, target_div, data_path) 
                               for num, data_path in enumerate(target_data_list)])
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    rets = None