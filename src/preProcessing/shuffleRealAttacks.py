from src.loadData import loadUsers
from numpy import zeros, ones, concatenate, arange
from random import random
from sklearn.model_selection import train_test_split

def getDataFromRaw(path = '../../users/' ):
    user_list, attack_list = loadUsers.readRealAndRandomUsers(path)
    return user_list, attack_list

def shuffleData(user_list, attack_list):
    x = []
    new_x = []
    new_y = []

    user_len = len(user_list)
    attack_len = len(attack_list)

    x.append(user_list)
    x.append(attack_list)

    y = concatenate((zeros(user_len), ones(attack_len)))

    index = arange(0, len(y))
    random(10).shuffle(index)

    for i in index:
        new_x.append(x[i])
        new_y.append(y[i])

    return new_x, new_y


def createTrainAndTestData():
    user_list, attack_list = getDataFromRaw()
    x, y = shuffleData(user_list, attack_list)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

    return x_train, x_test, y_train, y_test


