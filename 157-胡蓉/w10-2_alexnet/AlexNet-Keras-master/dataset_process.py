import os

photos = os.listdir("./data/image/train/")

# 该部分用于根据图片名信息生成图片标签对应文本
with open("ta/dataset.txt","w") as f:
    for photo in photos:
        name = photo.split(".")[0]
        if name=="cat":
            f.write(photo + ";0\n")
        elif name=="dog":
            f.write(photo + ";1\n")
f.close()