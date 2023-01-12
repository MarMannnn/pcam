def DataLoader():
    import gzip
    import shutil
    import h5py
    import numpy as np
    import gdown
    
    # train
    xt = 'https://drive.google.com/uc?export=download&id=1Ka0XfEMiwgCYPdTI-vv6eUElOBnKFKQ2'
    o_xt = 'train_x.h5.gz'
    gdown.download(xt, o_xt, quiet=False)

    yt = 'https://drive.google.com/uc?export=download&id=1269yhu3pZDP8UYFQs-NYs3FPwuK-nGSG'
    o_yt = 'train_y.h5.gz'
    gdown.download(yt, o_yt, quiet=False)
    
    # test
    xte = 'https://drive.google.com/uc?export=download&id=1qV65ZqZvWzuIVthK8eVDhIwrbnsJdbg_'
    o_xte = 'test_x.h5.gz'
    gdown.download(xte, o_xte, quiet=False)
    
    yte = 'https://drive.google.com/uc?export=download&id=17BHrSrwWKjYsOgTMmoqrIjDy6Fa2o_gP'
    o_yte = 'test_y.h5.gz'
    gdown.download(yte, o_yte, quiet=False)
     
    #valid
    xv = 'https://drive.google.com/uc?export=download&id=1hgshYGWK8V-eGRy8LToWJJgDU_rXWVJ3'
    o_xv = 'valid_x.h5.gz'
    gdown.download(xv, o_xv, quiet=False)
    
    yv = 'https://drive.google.com/uc?export=download&id=1bH8ZRbhSVAhScTS0p9-ZzGnX91cHT3uO'
    o_yv = 'valid_y.h5.gz'
    gdown.download(yte, o_yte, quiet=False)
    
    
    #unzip
    with gzip.open('train_x.h5.gz', 'rb') as f_in:
        with open('train_x.h5', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    with gzip.open('train_y.h5.gz', 'rb') as f_in:
        with open('train_y.h5', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    with gzip.open('test_x.h5.gz', 'rb') as f_in:
        with open('test_x.h5', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    with gzip.open('test_y.h5.gz', 'rb') as f_in:
        with open('test_y.h5', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
   
    with gzip.open('valid_x.h5.gz', 'rb') as f_in:
        with open('valid_x.h5', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
   
    with gzip.open('valid_y.h5.gz', 'rb') as f_in:
        with open('valid_y.h5', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)