import numpy as np
import cupy as cp
import time
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

from sklearn.externals import joblib

def SVM(x_data, x_label):
    from sklearn import model_selection, metrics
    from sklearn.svm import SVC
    X = cp.asnumpy(x_data)
    Y = cp.asnumpy(x_label)
    s = np.arange(np.array(Y).shape[0])
    np.random.shuffle(s)
    X = X[s]
    Y = Y[s]
    clf = SVC(kernel= 'rbf',gamma='scale', C= 15.0)
    test_size = 0.3
    seed = 7
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
    model = clf.fit(X_train, Y_train)
    filename = 'finalized_model_SVM.sav'
    joblib.dump(model, filename)
    loaded_model = joblib.load(filename)
    scores = cross_val_score(loaded_model, X, Y, cv=5)
    result = loaded_model.score(X_test, Y_test)
    pred= loaded_model.predict(X_test)
    print(np.average(scores), scores)
    print(result)
    print("훈련 세트 정확도: {:.4f}".format(loaded_model.score(X_train, Y_train)))
    print("테스트 세트 정확도: {:.4f}".format(loaded_model.score(X_test, Y_test)))
    print("훈련 세트 정확도:\n" ,metrics.confusion_matrix(loaded_model.predict(X_train), Y_train))
    print("테스트 세트 정확도:\n", metrics.confusion_matrix(loaded_model.predict(X_test), Y_test))
    np.savetxt('confusionmat_test.csv',  metrics.confusion_matrix(loaded_model.predict(X_test), Y_test), delimiter=',')
    np.savetxt('confusionmat_train.csv',  metrics.confusion_matrix(loaded_model.predict(X_train), Y_train), delimiter=',')

    return 0

def SVM(x_data, x_label):
    from sklearn import model_selection, metrics
    from sklearn.svm import SVC
    X = cp.asnumpy(x_data)
    Y = cp.asnumpy(x_label)
    s = np.arange(np.array(Y).shape[0])
    np.random.shuffle(s)
    X = X[s]
    Y = Y[s]
    clf = SVC(kernel= 'rbf',gamma='scale', C= 15.0)
    test_size = 0.3
    seed = 7
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
    model = clf.fit(X_train, Y_train)
    filename = 'finalized_model_SVM.sav'
    joblib.dump(model, filename)
    loaded_model = joblib.load(filename)
    scores = cross_val_score(loaded_model, X, Y, cv=5)
    result = loaded_model.score(X_test, Y_test)
    pred= loaded_model.predict(X_test)
    print(np.average(scores), scores)
    print(result)
    print("훈련 세트 정확도: {:.4f}".format(loaded_model.score(X_train, Y_train)))
    print("훈련 세트 정확도:\n" ,metrics.confusion_matrix(loaded_model.predict(X_train), Y_train))
    np.savetxt('confusionmat_train.csv',  metrics.confusion_matrix(loaded_model.predict(X_train), Y_train), delimiter=',')

    return 0