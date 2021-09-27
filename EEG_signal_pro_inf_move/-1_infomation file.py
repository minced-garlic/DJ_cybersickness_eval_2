#전혀 상관없는 파일입니다 그냥 파일 목록만 뽑아주는.


import os, shutil
pre_dir= 'D:\\2019임상데이터\\원본\\'

if __name__ == "__main__":
    filelist_learning = os.listdir(pre_dir)
    filelist_learning.sort()
    print(filelist_learning)