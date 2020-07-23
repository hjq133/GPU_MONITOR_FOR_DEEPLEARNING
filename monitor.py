import pynvml
import threading
import os
import time
import argparse

parser = argparse.ArgumentParser(description='GPU WATCHER')
parser.add_argument('--gpu_ids', type=str, default='0,1,2,3',
                    help='The id of the gpu to be monitored, separated by comma')
parser.add_argument('--proportion', type=str, default=8, help='Occupancy ratio threshold')
parser.add_argument('--time_threshold', type=int, default=600, help='Time threshold in seconds')
parser.add_argument('--command', type=str, default='python train_search.py --gpu=',  # we will add gpu automatically
                    help='Run your program with the corresponding gpu')
parser.add_argument('--ratio', type=float, default=1024 ** 2)
args = parser.parse_args()

pynvml.nvmlInit()
gpu_ids = list(map(int, args.gpu_ids.split(',')))
is_over = False  # Flag, if there exists a thread which is running your program, set is_over to True


class GpuMonitor(threading.Thread):
    def __init__(self, gpu_id):
        threading.Thread.__init__(self)
        self.time_threshold = args.time_threshold
        self.proportion = args.proportion
        self.ratio = args.ratio
        self.command = args.command + str(self.gpu_id)  # we will add gpu automatically

        self.gpu_id = gpu_id
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(id)
        self.memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.total = self.memory_info.total / self.ratio
        self.used = self.memory_info.used / self.ratio
        self.free = self.memory_info.used / self.ratio

    def run(self):  # running code
        if self.used < self.total / self.proportion:  # if we already have empty gpu
            if is_over:
                return
            self.run_program()
        else:
            print('gpu {} is being used, we are monitoring ...')
            while True:
                self.update_info()
                if self.used < self.total / self.proportion:
                    self.monitor()
                time.sleep(5)  # sleep for 5 seconds

    def update_info(self):
        self.memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
        self.total = self.memory_info.total / self.ratio
        self.used = self.memory_info.used / self.ratio
        self.free = self.memory_info.used / self.ratio

    def run_program(self):  # grabbed the gpu and run your program
        print('We successfully grabbed the gpu {}'.format(self.gpu_id))
        print('start run your program')
        global is_over
        is_over = True
        os.system(self.command)
        print('finish!')
        exit(0)

    def monitor(self):  # If no one is using it within 10 minutes, we will preempt the GPU
        for i in range(self.time_threshold):
            if self.used > self.total / self.proportion:
                return
            time.sleep(1)
        if is_over:
            return
        else:
            self.run_program()


def main():
    for _, gpu_id in enumerate(gpu_ids):
        GpuMonitor(args, gpu_id).start()
        time.sleep(1)


if __name__ == '__main__':
    main()
