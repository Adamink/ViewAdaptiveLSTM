from tqdm import tqdm
import numpy as np
import os
def reshape(data):
    # data: N C V T M
    # (37920, 3, 300, 25, 2) -> (37920, 300, 2 * 25 * 3)
    data = np.transpose(data,(0,2,4,3,1))
    batch = data.shape[0]
    time = data.shape[1]
    data = np.reshape(data, (batch, time, -1))
    return data
def strans(data,num_frame, two_person):
    # data: (batch, 800, 150)
    # num_frame: (batch, )
    # two_person: (batch, )
    print("==> strans")
    batch = data.shape[0]
    time = data.shape[1]
    feature_len = data.shape[2]
    data = np.reshape(data, (batch, time, -1, 3))
    origin = data[:,0,1,:]
    data = data - origin[:,None,None,:]
    data = np.reshape(data, (batch,time, -1))
    for i in tqdm(range(batch)):
        data[i, num_frame[i]:,:] = 0.
        if not two_person[i]:
            data[i,:, feature_len//2:] = 0.
    return data
def ftrans(data,num_frame, two_person):
    # data: (batch, 800, 150)
    # num_frame: (batch, )
    # two_person: (batch, )
    print("==> ftrans")
    batch = data.shape[0]
    time = data.shape[1]
    feature_len = data.shape[2]
    data = np.reshape(data, (batch, time, -1, 3))
    origin = data[:,:,1,:]
    data = data - origin[:,:,None,:]
    data = np.reshape(data, (batch, time, -1))
    for i in tqdm(range(batch)):
        data[i, num_frame[i]:,:] = 0.
        if not two_person[i]:
            data[i,:, feature_len//2:] = 0.
    return data
if __name__ == '__main__':
    benchmark = ['train','val']
    part = ['L','M','R','CS']
    trans = ['strans']
    #folder = '/mnt/data1/wuxiao/PKUMMD/npy_v2/'
    from global_variable import v2_npy_fd as folder
    for b in benchmark:
        print("==> benchmark: " + b)
        for p in part:
            print("==> part: " + p)
            for t in trans:
                in_path = os.path.join(folder, p)
                in_file_name = b + "_data_raw.npy"
                out_file_name = b + "_data_" + t + ".npy" 
                num_frame_file_name = b + "_num_frame.npy"
                two_person_file_name = b + "_two_person.npy"

                in_file_path = os.path.join(in_path, in_file_name)
                out_file_path = os.path.join(in_path, out_file_name)
                num_frame_file_path = os.path.join(in_path, num_frame_file_name)
                two_person_file_path = os.path.join(in_path, two_person_file_name)

                data = np.load(in_file_path)
                num_frame = np.load(num_frame_file_path)
                two_person = np.load(two_person_file_path)

                if t=='strans':
                    data = strans(data, num_frame, two_person)
                else:
                    data = ftrans(data, num_frame, two_person)
                np.save(out_file_path, data)

