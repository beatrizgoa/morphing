import os
import cv2
import numpy as np
import random
path = '../users'

def splitString(string, symbol):
    # string is the word that want to be splited and symbol the '.' o '_'
    return string.split(symbol)


def realOrAttack(user_number, second_image_number):
    if user_number == second_image_number:
        aux = 'real'
    else:
        aux = 'attack'

    return aux


def readRealImage(image_folder_path, user_numer):
    image_name = 'USER_'+user_numer+'_'+user_numer
    image_path = os.path.join(image_folder_path, image_name)

    return cv2.imread(image_path)


def getRandomUsers(user_numer):
    # This function gets two random users
    x1 = int(user_numer)
    x2 = int(user_numer)

    while x1 == user_numer and x2 == user_numer and x1 == x2:
        x1 = random.randint(0,185)
        x2 = random. randint(0,185)

    return x1, x2


def readRealAndRandomUsers(path_in):
    attack_list = []
    user_list = []

    # This function read real users and two random attacks users
    # For loop to read folder USER_XXX
    for pos, user in enumerate(os.listdir(path_in)):
        image_folder_path = os.path.join(path_in, user)
        user_number = splitString(user, ('_'))[1]

        # For each folder: read real user and the two random users
        real_user_image = readRealImage(image_folder_path, user_number)

        # get random users numbers
        x1, x2 = getRandomUsers(user_number)

        # get random users paths
        random1_path =  os.path.join(image_folder_path, 'USER_'+user_number+'_'+str(x1))
        random2_path = os.path.join(image_folder_path, 'USER_' + user_number + '_' + str(x2))

        random1 = cv2.imread(random1_path)
        random2 = cv2.imread(random2_path)

        # np ravel
        real_user_image = np.ravel(real_user_image)
        random1 = np.ravel(random1)
        random2 = np.ravel(random2)

        user_list.append(real_user_image)
        attack_list.append(random1)
        attack_list.append(random2)

    return user_list, attack_list


def readAllUsers(path_in):

    # For loop to read folder USER_XXX
    for pos, user in enumerate(os.listdir(path_in)):

        user_list = []
        attack_list = []

        image_folder_path = os.path.join(path_in, user)
        user_number = splitString(user,('_'))[1]

        # For loop to read image USER_XXX_XXX in USER_XXX folder
        for image in os.listdir(image_folder_path):
            image_path = os.path.join(image_folder_path, image)

            image_name = splitString(image,('.'))[0] # get the non .jpg part
            second_image_number = splitString(image_name, ('_'))[2] # get the user number


            image_class = realOrAttack(user_number, second_image_number)
            print( user_number, second_image_number, image_class)

            print ('-------------------------------------------')

            #Read image and resize it
            image = cv2.imread(image_path)
            image = np.ravel(image)

            if image_class == 'real':
                user_list.append(image)

            else:
                attack_list.append(image)

    return user_list, attack_list

