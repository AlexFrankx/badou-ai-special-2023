原本文件vgg_16.ckpt有500多M,github无法上传
先将vgg_16.ckpt压缩为vgg_16.zip，再执行命令拆分
split -b 100M -d vgg_16.zip vgg16.zip.

$ ll -h
total 1.5G
-rw-r--r-- 1 胡蓉 197121 100M Feb  2 19:42 vgg16.zip.00
-rw-r--r-- 1 胡蓉 197121 100M Feb  2 19:42 vgg16.zip.01
-rw-r--r-- 1 胡蓉 197121 100M Feb  2 19:42 vgg16.zip.02
-rw-r--r-- 1 胡蓉 197121 100M Feb  2 19:42 vgg16.zip.03
-rw-r--r-- 1 胡蓉 197121  90M Feb  2 19:42 vgg16.zip.04
-rw-r--r-- 1 胡蓉 197121 528M Feb  2 18:30 vgg_16.ckpt
-rw-r--r-- 1 胡蓉 197121 490M Feb  2 19:34 vgg_16.zip
