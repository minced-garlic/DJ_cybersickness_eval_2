
import numpy as np
import os

'''
: dataset_dir= dataset가 저장된 폴더
: train , test= train 과 test의 비율
: testing = 데이터를 소수로 불러와 제대로 작동하는지 확인하기 위함
: info= 각 데이터별 인포를 함께불러와 어느 데이터가 분류가 안되는지 확인하기 위함
: scores= 각 스코어 별 
'''

aver_dir = 'D:\\2019임상데이터\\원본\\experimental_data\\'
def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False

def dataset(dataset_dir, train=4, test=1, testing= False, info= False, scores= None, ):
    dataset_X, dataset_y, dataset_inf, EEG_data, count_score = [], [], [], [],[0,0,0,0,0,0]
    filelist = os.listdir(dataset_dir)
    filelist.sort()

    ###testing일 시 1000개의 데이터만 가지고 하도록###
    if testing:
        linelen= 100
    else:
        linelen=len(filelist)

    ###데이터 입력단###
    if 'csv' in filelist[0]:
        for line in range(linelen):
            data = np.loadtxt(dataset_dir + filelist[line], delimiter=',')
            EEG_data.append(data)
            if line == len(filelist) - 1 or filelist[line][:-8] != filelist[line + 1][:-8]:
                EEG_data = np.transpose(EEG_data, (1, 2, 0))
                if len(dataset_X) > 1 and np.array(dataset_X[len(dataset_X) - 1]).shape != np.array(EEG_data).shape:
                    print(len(dataset_X) - 1, '번 데이터가 다름:', np.array(EEG_data).shape, filelist[line])
                else:
                    score= int(filelist[line][-5]) - 1
                    dataset_X.append(EEG_data)
                    dataset_y.append(score)
                    count_score[score]+=1
                EEG_data = []
    elif 'npy' in filelist[0]:
        for line in range(linelen):
            score= int(filelist[line][-5])-1
            EEG_data=np.load(dataset_dir + filelist[line])
            EEG_data = np.transpose(EEG_data, (1, 2, 0))
            dataset_X.append(EEG_data)
            dataset_y.append(score)
            count_score[score]+=1

    else:
        print(filelist[0], 'is not datafile format')

    ### 데이터 셔플
    dataset_X = np.array(dataset_X)
    dataset_y = np.array(dataset_y)
    number = dataset_X.shape[0]
    s = np.arange(np.array(dataset_y).shape[0])
    np.random.shuffle(s)
    dataset_X,dataset_y= dataset_X[s],dataset_y[s]

    if info:
        for line in range(linelen):
            dataset_inf.append([filelist[line][5:8], filelist[line][18:22]])
        dataset_inf = np.array(dataset_inf)
        dataset_inf =dataset_inf[s]
    else:
        dataset_inf = None
    if not scores is None:
        dataset_y= data_score(dataset_y, scores)
    print('score분포:', count_score)
    rate = int(number * test / (train + test))
    return dataset_X[rate:], dataset_X[:rate], dataset_y[rate:], dataset_y[:rate], dataset_inf


# dataset_label_n_score는 직접 스코어를 학습하는 게 아니라 평균스코어와 라벨링을 따로함.
# input 똑같고 output이 두배.

def dataset_label_n_score(dataset_dir, train=1, test=1, testing=False):
    person_aver ,scene_aver= data_average(aver_dir)

    dataset_X, dataset_ps,dataset_scene,dataset_score, EEG_data = [], [],[] ,[],[]
    filelist = os.listdir(dataset_dir)
    filelist.sort()

    ###testing일 시 100개의 데이터만 가지고 하도록###
    if testing:
        linelen= 100
    else:
        linelen=len(filelist)

    if 'npy' in filelist[0]:
        for line in range(linelen):
            score= int(filelist[line][-5]) - 1
            EEG_data=np.load(dataset_dir + filelist[line])
            EEG_data = np.transpose(EEG_data, (1, 2, 0))
            dataset_X.append(EEG_data)
            #print(np.random.normal(person_aver[filelist[line][5:8]][0], person_aver[filelist[line][5:8]][1], 1))
            dataset_ps.append(np.random.normal(person_aver[filelist[line][5:8]][0], person_aver[filelist[line][5:8]][1], 1)[0])
            dataset_scene.append(np.random.normal(scene_aver[filelist[line][18:22]][0], scene_aver[filelist[line][18:22]][1], 1)[0])
            dataset_score.append(score)
        ### 데이터 셔플
        dataset_X = np.array(dataset_X)
        dataset_ps = np.array(dataset_ps)
        dataset_scene = np.array(dataset_scene)
        dataset_score=np.array(dataset_score)
        number = dataset_X.shape[0]
        s = np.arange(np.array(dataset_score).shape[0])
        np.random.shuffle(s)
        dataset_X,dataset_ps,dataset_scene, dataset_score= dataset_X[s],dataset_ps[s],dataset_scene[s],dataset_score[s]

        rate = int(number * test / (train + test))
        return dataset_X[rate:], dataset_X[:rate], dataset_ps[rate:], dataset_ps[:rate]\
            ,dataset_scene[rate:], dataset_scene[:rate],dataset_score[:rate],


    else:
        print(filelist[0], 'is not datafile format')

def dataset_label_n_score2(dataset_dir, train=4, test=1,  testing=False):
    person_aver ,scene_aver= data_average2(aver_dir)

    dataset_X, dataset_ps,dataset_scene,dataset_score, EEG_data = [], [],[] ,[],[]
    filelist = os.listdir(dataset_dir)
    filelist.sort()

    ###testing일 시 1000개의 데이터만 가지고 하도록###
    if testing:
        linelen= 200
    else:
        linelen=len(filelist)

    if 'npy' in filelist[0]:
        for line in range(linelen):
            score= int(filelist[line][-5]) - 1
            EEG_data=np.load(dataset_dir + filelist[line])
            EEG_data = np.transpose(EEG_data, (1, 2, 0))
            dataset_X.append(EEG_data)
            #print(np.random.normal(person_aver[filelist[line][5:8]][0], person_aver[filelist[line][5:8]][1], 1))
            dataset_ps.append(person_aver[filelist[line][5:8]])
            dataset_scene.append(scene_aver[filelist[line][18:22]])
            dataset_score.append(score)
        ### 데이터 셔플
        dataset_X = np.array(dataset_X)
        dataset_ps = np.array(dataset_ps)
        dataset_scene = np.array(dataset_scene)
        dataset_score=np.array(dataset_score)
        number = dataset_X.shape[0]
        s = np.arange(np.array(dataset_score).shape[0])
        np.random.shuffle(s)
        dataset_X,dataset_ps,dataset_scene, dataset_score= dataset_X[s],dataset_ps[s],dataset_scene[s],dataset_score[s]

        rate = int(number * test / (train + test))
        return dataset_X[rate:], dataset_X[:rate], dataset_ps[rate:], dataset_ps[:rate]\
            ,dataset_scene[rate:], dataset_scene[:rate],dataset_score[rate:], dataset_score[:rate], [dataset_ps.max(),dataset_scene.max()]


    else:
        print(filelist[0], 'is not datafile format')


def data_average2(aver_dir):
    person_aver, scene_aver = {}, {}
    filelist = os.listdir(aver_dir)
    filelist.sort()
    linelen = len(filelist)
    scene_num=0
    for file_num in range(linelen):
        person_aver[filelist[file_num][5:8]] = int(filelist[file_num][5:8])
        score_dir = os.listdir(aver_dir + filelist[file_num] + '/' + filelist[file_num] + '/SSQ/')
        score_dir.sort()
        data = np.loadtxt(aver_dir + filelist[file_num] + '/' + filelist[file_num] + '/SSQ/' + score_dir[1],
                          delimiter=',', dtype='str')
        key=filelist[file_num][5:8]
        person_aver[key]=[]
        for data in data:
            if isNumber(data[0]):
                if scene_aver.get(data[1]) is None:
                    scene_aver[data[1]] = scene_num
                    scene_num+=1
                person_aver[key].append(float(data[0]))
        person_aver[key] = np.array(person_aver[key])
        person_aver[key] = np.mean(person_aver[key])

    person_min=min(person_aver.values())
    person_max=max(person_aver.values())

    for key in person_aver:
        person_aver[key] -=person_min
        person_aver[key] /=(person_max-person_min)
        person_aver[key] *=10
        person_aver[key]=int(person_aver[key])

    np.save('./person_class.npy', person_aver)
    np.save('./scene_class.npy', scene_aver)
    return person_aver, scene_aver

def data_average(aver_dir):
    person_aver, scene_aver = {},{}
    filelist = os.listdir(aver_dir)
    filelist.sort()
    linelen=len(filelist)
    for file_num in range(linelen):
        person_aver[filelist[file_num][5:8]] =[]
        score_dir = os.listdir(aver_dir + filelist[file_num]+'/'+filelist[file_num]+'/SSQ/')
        score_dir.sort()
        data = np.loadtxt(aver_dir + filelist[file_num]+'/'+filelist[file_num]+'/SSQ/'+score_dir[1], delimiter=',',dtype='str')
        for data in data:
            if isNumber(data[0]):
                score= float(data[0])
                if scene_aver.get(data[1]) is None:
                    scene_aver[data[1]]=[score]
                else:
                    scene_aver[data[1]].append(score)
                person_aver[filelist[file_num][5:8]].append(score)
    for key in person_aver:
        person_aver[key] = np.array(person_aver[key])
        person_aver[key] = [np.mean(person_aver[key]),np.std(person_aver[key])]

    for key in scene_aver:
        scene_aver[key] = np.array(scene_aver[key])
        scene_aver[key] = [np.mean(scene_aver[key]),np.std(scene_aver[key])]

    np.save('./person_aver.npy',person_aver)
    np.save('./scene_aver.npy',scene_aver)
    return person_aver, scene_aver





def data_score(dataset_y, scores):
    dataset_y=[scores[data_y] for data_y in dataset_y]
    return np.array(dataset_y)




if __name__ == "__main__":
    aver_dir= 'D:\\2019임상데이터\\원본\\experimental_data\\'
    data_average(aver_dir)