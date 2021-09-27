
import numpy as np
import os
pre_dir= 'D:\\2019_data_for_Cnn\\rest_raw\\'
post_dir= 'D:\\2019_data_for_Cnn\\rest_infoICA\\'

from ica import ICA
def ICAeeg(pre_dir, post_dir):
    S= np.load(pre_dir)[:-2]

    myICA = ICA(n_components=14, method='infomax')
    myICA.fit(S)
    s2 = myICA.transform(S)

    #np.savetxt(post_dir, s2, delimiter=',')
    np.save(post_dir,s2)




if __name__ == "__main__":
    filelist_learning = os.listdir(pre_dir)
    filelist_learning.sort()

    for filelist in filelist_learning :
        if not os.path.isfile(post_dir+filelist):
            ICAeeg(pre_dir+filelist, post_dir+filelist)
            print(filelist,' success')