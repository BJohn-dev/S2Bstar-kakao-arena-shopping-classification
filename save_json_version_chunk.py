import os
import sys
import json
import pickle as pkl
import time
from contextlib import contextmanager
from multiprocessing import Pool

import numpy as np
import h5py


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def save_chunk(div_and_path):
    
    num, div, data_path = div_and_path
    print(div_and_path)
    with timer("Parse chunk {}".format(num)):   
        train_h = h5py.File(data_path, 'r')
        train = train_h[div]

        with open(os.path.join('data/json_version_chunk/', "json_chunck.{}.{}.json".format(div,num)), 'w') as f:
            
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
                        # Leave log~!!!!!
                        break
                    
                    key = str(key)
                    line_dict[key] = value
                
                f.write(json.dumps(line_dict))
                f.write("\n")
                
                #############################################
                #############################################
                # break
                #############################################
                #############################################
    return


if __name__ == "__main__":
   
    # div = "train"
    train_data_list = [
            "../train.chunk.01",
            "../train.chunk.02",
            "../train.chunk.03",
            "../train.chunk.04",
            "../train.chunk.05",
            "../train.chunk.06",
            "../train.chunk.07",
            "../train.chunk.08",
            "../train.chunk.09"
    ]
    num_workers = len(train_data_list)
    save_dir = 'data/json_version_chunk/' 

    # div = "dev"
    dev_data_list = ["../dev.chunk.01"]
    
    # div = "test"
    test_data_list = ["../test.chunk.01",
                      "../test.chunk.02"]
    
    
    target_div = sys.argv[1]
    target_data_list = None
    if target_div == 'dev':
        target_data_list = dev_data_list
    elif target_div == 'test':
        target_data_list = test_data_list
    elif target_div == 'train':
        target_data_list = train_data_list

    print(target_div, len(target_data_list))
    if os.path.isdir(save_dir):
        pass
    else:
        os.mkdir(save_dir)

    pool = Pool(num_workers)
    try:
        rets = pool.map_async(save_chunk, [(num, target_div, data_path) for num, data_path in enumerate(target_data_list)])
        pool.close()
        pool.join()

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
    rets = None