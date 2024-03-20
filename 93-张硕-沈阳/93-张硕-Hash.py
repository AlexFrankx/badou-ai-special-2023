import cv2
import numpy as np

def aHash(img):
    img = cv2.resize(img,(8,8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    s = 0
    hash_str = ''

    for i in range(8):
        for j in range(8):
            s = s + gray[i, j]
    avg = s / 64
    for i in range(8):
        for j in range(8):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str


def dHash(img):
    img = cv2.resize(img, (9, 8), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    for i in range(8):
        for j in range(8):
            if gray[i, j] > gray[i, i+1]:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

#汉明距离
def cmpHash(hash1, hash2):
    n = 0
    if len(hash1) != len(hash2):
        return -1

    for i in range(len(hash1)):
        if hash1[i] != hash2[i]:
            n = n + 1
    return n

