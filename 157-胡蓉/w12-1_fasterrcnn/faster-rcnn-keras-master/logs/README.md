训练好的文件会存放在这里
进入虚拟环境后conda activate badou
执行 Tensorboard --logdir=logs的绝对路径，加引号，可出现一个tensorboard的网址，点击进去可可视化网络情况
如果 Tensorboard报错，可以用一下方式执行
python -m tensorboard.main --logdir=logs

tensorflow 2.10.1的结果没法可视化，可能是版本问题导致的