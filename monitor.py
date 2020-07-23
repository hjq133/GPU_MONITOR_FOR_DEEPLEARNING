import pynvml
import threading
import os
import time

pynvml.nvmlInit()
gpu_ids = [0, 1, 2, 3]  # gpu id
gpu_handle = [pynvml.nvmlDeviceGetHandleByIndex(id) for _, id in enumerate(gpu_ids)]
meminfo = [0 for _, id in enumerate(gpu_ids)]
total = [0 for _, id in enumerate(gpu_ids)]
used = [0 for _, id in enumerate(gpu_ids)]
free = [0 for _, id in enumerate(gpu_ids)]
ratio = 1024 ** 2
proportion = 8.0


# class GpuMonitor(threading.Thread):  # 继承父类threading.Thread
#     def __init__(self, gpu_id):
#         threading.Thread.__init__(self)
#         self.gpu_id = gpu_id
#
#     def run(self):  # 把要执行的代码写到run函数里面 线程在创建后会直接运行run函数
#         os.system('python train.py')
#         print("finish")

def monitor(gpu_id, handle):  # 监测gpu是否已经长时间无人使用
    for i in range(600):  # 循环10分钟
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total = meminfo.total / ratio  # 以兆M为单位就需要除以1024**2
        used = meminfo.used / ratio
        if used > total / proportion:
            return
        time.sleep(1)
    print("now grabbed the gpu{}".format(gpu_id))
    command = "python train_search.py --gpu={}".format(gpu_id)
    os.system(command)
    print("now grabbed the gpu{} finish!".format(gpu_id))
    exit()


def main():
    while 1:
        for i, handle in enumerate(gpu_handle):
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total = meminfo.total / ratio  # 以兆M为单位就需要除以1024**2
            used = meminfo.used / ratio
            free = meminfo.free / ratio
            print("gpu id:{} | total :{} | used: {} | free: {}".format(gpu_ids[i], total, used, free))
            if used < total / proportion:
                print("start monitor gpu : {}".format(gpu_ids[i]))
                monitor(gpu_ids[i], handle)
                break
        print("-" * 89)
        time.sleep(1)  # 每隔1s监测一次gpu的状况


if __name__ == '__main__':
    main()
