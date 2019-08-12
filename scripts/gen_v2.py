import os
import numpy as np
from tqdm import tqdm
from glob import glob

cs_valid_names = "A01N02-L, A01N02-M, A01N02-R, A01N03-L, A01N03-M, A01N03-R, A01N07-L, A01N07-M, A01N07-R, A02N02-L, A02N02-M, A02N02-R, A02N03-L, A02N03-M, A02N03-R, A02N07-L, A02N07-M, A02N07-R, A03N02-L, A03N02-M, A03N02-R, A03N03-L, A03N03-M, A03N03-R, A03N07-L, A03N07-M, A03N07-R, A04N02-L, A04N02-M, A04N02-R, A04N03-L, A04N03-M, A04N03-R, A04N07-L, A04N07-M, A04N07-R, A05N02-L, A05N02-M, A05N02-R, A05N03-L, A05N03-M, A05N03-R, A05N07-L, A05N07-M, A05N07-R, A06N02-L, A06N02-M, A06N02-R, A06N03-L, A06N03-M, A06N03-R, A06N07-L, A06N07-M, A06N07-R, A07N02-L, A07N02-M, A07N02-R, A07N03-L, A07N03-M, A07N03-R, A07N07-L, A07N07-M, A07N07-R, A08N02-L, A08N02-M, A08N02-R, A08N03-L, A08N03-M, A08N03-R, A08N07-L, A08N07-M, A08N07-R, A09N02-L, A09N02-M, A09N02-R, A09N03-L, A09N03-M, A09N03-R, A09N07-L, A09N07-M, A09N07-R, A10N02-L, A10N02-M, A10N02-R, A10N03-L, A10N03-M, A10N03-R, A10N07-L, A10N07-M, A10N07-R, A11N02-L, A11N02-M, A11N02-R, A11N03-L, A11N03-M, A11N03-R, A11N07-L, A11N07-M, A11N07-R, A12N02-L, A12N02-M, A12N02-R, A12N03-L, A12N03-M, A12N03-R, A12N07-L, A12N07-M, A12N07-R, A13N02-L, A13N02-M, A13N02-R, A13N03-L, A13N03-M, A13N03-R, A13N07-L, A13N07-M, A13N07-R, A14N02-L, A14N02-M, A14N02-R, A14N03-L, A14N03-M, A14N03-R, A14N07-L, A14N07-M, A14N07-R, A15N02-L, A15N02-M, A15N02-R, A15N03-L, A15N03-M, A15N03-R, A15N07-L, A15N07-M, A15N07-R, A16N02-L, A16N02-M, A16N02-R, A16N03-L, A16N03-M, A16N03-R, A16N07-L, A16N07-M, A16N07-R, A17N02-L, A17N02-M, A17N02-R, A17N03-L, A17N03-M, A17N03-R, A17N07-L, A17N07-M, A17N07-R, A18N02-L, A18N02-M, A18N02-R, A18N03-L, A18N03-M, A18N03-R, A18N07-L, A18N07-M, A18N07-R, A19N02-L, A19N02-M, A19N02-R, A19N03-L, A19N03-M, A19N03-R, A19N07-L, A19N07-M, A19N07-R, A20N02-L, A20N02-M, A20N02-R, A20N03-L, A20N03-M, A20N03-R, A20N07-L, A20N07-M, A20N07-R, A21N02-L, A21N02-M, A21N02-R, A21N03-L, A21N03-M, A21N03-R, A21N07-L, A21N07-M, A21N07-R, A22N02-L, A22N02-M, A22N02-R, A22N03-L, A22N03-M, A22N03-R, A22N07-L, A22N07-M, A22N07-R, A23N02-L, A23N02-M, A23N02-R, A23N03-L, A23N03-M, A23N03-R, A23N07-L, A23N07-M, A23N07-R, A24N02-L, A24N02-M, A24N02-R, A24N03-L, A24N03-M, A24N03-R, A24N07-L, A24N07-M, A24N07-R, A25N02-L, A25N02-M, A25N02-R, A25N03-L, A25N03-M, A25N03-R, A25N07-L, A25N07-M, A25N07-R, A26N02-L, A26N02-M, A26N02-R, A26N03-L, A26N03-M, A26N03-R, A26N07-L, A26N07-M, A26N07-R"
cs_valid_names = [x.strip() for x in cs_valid_names.split(",")]

def isTrain(pth, setting):
    file_name = os.path.basename(pth).split('.')[0]
    if setting!='CS':
        if file_name[-1] in setting:
            return False
        return True
    else:
        if file_name in cs_valid_names:
            return False
        return True
    
if __name__=='__main__':
    # data_fd = '/mnt/data1/wuxiao/PKUMMD/skeleton_v2/'
    # label_fd= '/mnt/data1/wuxiao/PKUMMD/label_v2/'
    # dst_fd = '/mnt/data1/wuxiao/PKUMMD/npy_v2/'
    from global_variable import v2_data_fd as data_fd
    from global_variable import v2_label_fd as label_fd
    from global_variable import v2_npy_fd as dst_fd
    settings = ['L', 'M', 'R', 'CS']
    two_person_label = [12,14,16,18,21,24,26,27,29,38]

    label_pths = glob(os.path.join(label_fd, '*'))
    # data_pths = [s.replace('label_v2','skeleton_v2') for s in label_pths]
    data_pths = glob(os.path.join(data_fd), '*')

    for setting in settings:
        print("==> " + setting)
        fd = os.path.join(dst_fd, setting)
        if not os.path.exists(fd):
            os.makedirs(fd)

        train_line_cnt = 0
        val_line_cnt = 0
        train_labels = []
        train_starts = []
        val_labels = []
        for i in range(len(label_pths)):
            label_file = open(label_pths[i],'r') 
            lines = label_file.readlines()
            for line in lines:
                if line.strip():
                    if isTrain(label_pths[i], setting):
                        train_line_cnt += 1
                        train_labels.append(int(line.strip().split(',')[0]))
                        train_starts.append(int(line.strip().split(',')[1]))
                    else:
                        val_line_cnt +=1
                        val_labels.append(int(line.strip().split(',')[0]))
            label_file.close()
        # print(min(train_labels)) 1
        # print(max(train_labels)) 51
        # print(min(train_starts)) 0

        train_data_numpy = np.zeros(shape = (train_line_cnt, 800, 150))
        val_data_numpy = np.zeros(shape = (val_line_cnt, 800, 150))
        train_label_numpy = np.zeros(shape= (train_line_cnt,),dtype=int)
        val_label_numpy = np.zeros(shape= (val_line_cnt,),dtype=int)
        train_num_frame_numpy = np.zeros(shape = (train_line_cnt,), dtype=int)
        val_num_frame_numpy = np.zeros(shape = (val_line_cnt,), dtype=int)
        train_two_person_numpy = np.zeros(shape = (train_line_cnt, ), dtype=bool)
        val_two_person_numpy = np.zeros(shape = (val_line_cnt, ), dtype=bool)

        train_it = 0
        val_it = 0

        for i in tqdm(range(len(label_pths))):
            pth = label_pths[i]
            with open(label_pths[i],'r') as label_file:
                label_lines = label_file.readlines()
            with open(data_pths[i],'r') as data_file:
                data_lines = data_file.readlines()
            if isTrain(pth, setting):
                for line in label_lines:
                    if line.strip():
                        try:
                            label,start,end = [int(x) for x in line.strip().split(',')][:3]
                        except:
                            print("train:")
                            print("label_pth\n" + str(pth))
                            print("data_pth\n" + data_pths[i])
                            print("line: " + str(line))
                            exit()
                        t = 0
                        for t,j in enumerate(range(start, end)):
                            try:
                                line = data_lines[j].strip().split()
                            except:
                                print("train:")
                                print("label_pth\n" + str(pth))
                                print("data_pth\n" + data_pths[i])
                                print("j=" + str(j))
                                print("len(data_lines)=" + str(len(data_lines)))
                                exit()
                            for k in range(150):
                                try:
                                    train_data_numpy[train_it, t, k] = float(line[k])
                                except:
                                    print(line)
                                    print(line[k])
                                    exit()
                        train_num_frame_numpy[train_it] = end - start
                        train_label_numpy[train_it] = label
                        train_two_person_numpy[train_it] = True if label in two_person_label else False
                        train_it += 1
            else:
                for line in label_lines:
                    if line.strip():
                        label,start,end = [int(x) for x in line.strip().split(',')][:3]
                        t = 0
                        for t,j in enumerate(range(start, end)):
                            try:
                                line = data_lines[j].strip().split()
                            except:
                                print("test:")
                                print("pth\n" + str(pth))
                                print("j=" + str(j))
                                print("len(data_lines)=" + str(len(data_lines)))
                                exit()
                            for k in range(150):
                                try:
                                    val_data_numpy[val_it, t, k] = float(line[k])
                                except:
                                    print(line)
                                    print(line[k])
                                    exit()
                        val_num_frame_numpy[val_it] = end - start
                        val_label_numpy[val_it] = label
                        val_two_person_numpy[val_it] = True if label in two_person_label else False
                        val_it += 1
        
        print("==> saving data")
        np.save(os.path.join(fd, 'train_data_raw.npy'), train_data_numpy)
        np.save(os.path.join(fd, 'train_num_frame'), train_num_frame_numpy)
        np.save(os.path.join(fd, 'train_two_person'), train_two_person_numpy)
        np.save(os.path.join(fd, 'train_label'), train_label_numpy)
        np.save(os.path.join(fd, 'val_data_raw.npy'), val_data_numpy)
        np.save(os.path.join(fd, 'val_num_frame'), val_num_frame_numpy)
        np.save(os.path.join(fd, 'val_two_person'), val_two_person_numpy)
        np.save(os.path.join(fd, 'val_label'), val_label_numpy)


