import os
import numpy as np
from tqdm import tqdm
from glob import glob

cs_valid_names = "0291-L, 0291-M, 0291-R, 0292-L, 0292-M, 0292-R, 0293-L, 0293-M, 0293-R, 0294-L, 0294-M, 0294-R, 0295-L, 0295-M, 0295-R, 0296-L, 0296-M, 0296-R, 0297-L, 0297-M, 0297-R, 0298-L, 0298-M, 0298-R, 0299-L, 0299-M, 0299-R, 0300-L, 0300-M, 0300-R, 0301-L, 0301-M, 0301-R, 0302-L, 0302-M, 0302-R, 0303-L, 0303-M, 0303-R, 0304-L, 0304-M, 0304-R, 0305-L, 0305-M, 0305-R, 0306-L, 0306-M, 0306-R, 0307-L, 0307-M, 0307-R, 0308-L, 0308-M, 0308-R, 0309-L, 0309-M, 0309-R, 0310-L, 0310-M, 0310-R, 0311-L, 0311-M, 0311-R, 0312-L, 0312-M, 0312-R, 0313-L, 0313-M, 0313-R, 0314-L, 0314-M, 0314-R, 0315-L, 0315-M, 0315-R, 0316-L, 0316-M, 0316-R, 0317-L, 0317-M, 0317-R, 0318-L, 0318-M, 0318-R, 0319-L, 0319-M, 0319-R, 0320-L, 0320-M, 0320-R, 0321-L, 0321-M, 0321-R, 0322-L, 0322-M, 0322-R, 0323-L, 0323-M, 0323-R, 0324-L, 0324-M, 0324-R, 0325-L, 0325-M, 0325-R, 0326-L, 0326-M, 0326-R, 0327-L, 0327-M, 0327-R, 0328-L, 0328-M, 0328-R, 0329-L, 0329-M, 0329-R, 0330-L, 0330-M, 0330-R, 0331-L, 0331-M, 0331-R, 0332-L, 0332-M, 0332-R, 0333-L, 0333-M, 0333-R, 0334-L, 0334-M, 0334-R"
cs_valid_names = [x.strip() for x in cs_valid_names.split(",")]

def isTrain(pth, setting):
    file_name = os.path.basename(pth).split('.')[0]
    if setting!='CS':
        if file_name[-1] in setting:
            return False
        return True
    else:
        if file_name[-6:] in cs_valid_names:
            return False
        return True

def get_num_frame(x):
    ''' 
    x: (batch, seq_len, feature_len)
    return (batch,)
    '''
    print("==> get num_frame")
    eps = 0.001
    batch = x.shape[0]
    seq_len = x.shape[1]
    ret = np.zeros(shape = (batch), dtype = int)
    for i in tqdm(range(batch)):
        find_zero = False
        for j in range(seq_len):
            if abs(np.sum(x[i,j])) < eps:
                find_zero = True
                ret[i] = j
                break
        if not find_zero:
            ret[i] =  seq_len
    return ret
def get_two_person(x):
    '''
    x: (batch, seq_len, feature_len)
    return (batch,), dtype = bool
    '''
    print("==> get_two_person")
    eps = 0.001
    batch = x.shape[0]
    seq_len = x.shape[1]
    feature_len = x.shape[2]
    ret = np.zeros(shape = (batch), dtype = bool)
    for i in tqdm(range(batch)):
        if abs(np.sum(x[i,:, feature_len//2:])) < eps:
            ret[i] = False
        else:
            ret[i] = True
    return ret

if __name__=='__main__':
    from global_variable import v1_data_fd as data_fd
    from global_variable import v1_npy_fd as dst_fd
    # data_fd = '/mnt/Action2/temp_PKUMMD_v1/skeleton_features/' # path on titan1
    # dst_fd = '/mnt/Action2/PKUMMD_v1/norm_input/'

    settings = ['L', 'M', 'R', 'CS']
    two_person_label = [12,14,16,18,21,24,26,27,29,38]

    # data_pths = [s.replace('label_v1','skeleton_v1') for s in label_pths]

    data_pths = glob(os.path.join(data_fd, '*'))
    for setting in settings:
        print("==> " + setting)
        fd = os.path.join(dst_fd, setting)
        if not os.path.exists(fd):
            os.makedirs(fd)

        train_data = []
        train_label = []
        val_data = []
        val_label = []


        for data_pth in tqdm(data_pths):
            label = (int)(os.path.basename(data_pth).split('_')[0]) # 38_00_0148-M.npy
            input_data = np.load(data_pth) # (188, 150) for example
            data = np.zeros(shape = (800, 150), dtype=np.float32)
            data[:len(input_data),:] = input_data

            if isTrain(data_pth, setting):
                train_data.append(data)
                train_label.append(label)
            else:
                val_data.append(data)
                val_label.append(label)
        
        train_data_numpy = np.zeros(shape = (len(train_data), 800, 150))
        val_data_numpy = np.zeros(shape = (len(val_data), 800, 150))
        for i in tqdm(range(len(train_data))):
            train_data_numpy[i,:len(train_data[i])] = train_data[i]
        for i in tqdm(range(len(val_data))):
            val_data_numpy[i,:len(val_data[i])] = val_data[i]
        
        train_label_numpy = np.asarray(train_label)
        val_label_numpy = np.asarray(val_label)
        
        train_num_frame_numpy = get_num_frame(train_data_numpy)
        val_num_frame_numpy = get_num_frame(val_data_numpy)

        train_two_person_numpy = get_two_person(train_data_numpy)
        val_two_person_numpy = get_two_person(val_data_numpy)
        
        print("==> saving data")
        np.save(os.path.join(fd, 'train_data_raw.npy'), train_data_numpy)
        np.save(os.path.join(fd, 'train_num_frame'), train_num_frame_numpy)
        np.save(os.path.join(fd, 'train_two_person'), train_two_person_numpy)
        np.save(os.path.join(fd, 'train_label'), train_label_numpy)
        np.save(os.path.join(fd, 'val_data_raw.npy'), val_data_numpy)
        np.save(os.path.join(fd, 'val_num_frame'), val_num_frame_numpy)
        np.save(os.path.join(fd, 'val_two_person'), val_two_person_numpy)
        np.save(os.path.join(fd, 'val_label'), val_label_numpy)


