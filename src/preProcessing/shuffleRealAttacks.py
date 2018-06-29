from src.loadData import loadUsers


def createDataAndTrain():
    user_list, attack_list = loadUsers.readRealAndRandomUsers('../../users/')

