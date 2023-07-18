import multiprocessing
import time
from process import process
from process import plotPolarColor
import datetime
import os
import socket
import pickle
import sys
import time


import csiread
import numpy as np

import matplotlib.pyplot as plt
import struct
import sys
import time


    
def process1(queue1):
    # 第一个进程，每1秒向队列中放入一个值
    SIZE_BUFFER = 3600
    HZ = 800
    TIME = 20000
    NUM_TOTAL = HZ * TIME

    socket.NETLINK_CONNECTOR = 11
    CN_NETLINK_USERS = 11
    CN_IDX_IWLAGN = CN_NETLINK_USERS + 0xF
    NETLINK_ADD_MEMBERSHIP = 1


    # --------------------------------- realtime --------------------------------- #


    L = 40  # fixed window length
    FPS_RATIO = 2  # eval per second
    FRAC = int(L / FPS_RATIO)

    Ntx = 1
    Nrx = 3
    Nsub = 30

    csi_L = np.zeros((L, Nsub, Nrx, Ntx))
    csi_frac = np.zeros((FRAC, Nsub, Nrx, Ntx))
    
    while True:

        csidata = csiread.Intel(None, Nrx, Ntx)
        count = 0

        with socket.socket(
            socket.AF_NETLINK, socket.SOCK_DGRAM, socket.NETLINK_CONNECTOR
        ) as s:
            s.bind((os.getpid(), CN_IDX_IWLAGN))
            s.setsockopt(270, NETLINK_ADD_MEMBERSHIP, CN_IDX_IWLAGN)

            while count < NUM_TOTAL:
                ret = s.recv(SIZE_BUFFER)
                # keep nothing but data part
                cnmsg_data = ret[36:]
                # parse data using csiread.Intel.pmsg
                # only requires data bytes, omitting any other bytes
                csi_i = None
                status = csidata.pmsg(cnmsg_data)
                # csi from one packet
                if status == 0xBB:
                    # 187, status valid, else error
                    csi_i = csidata.get_scaled_csi()
                    count += 1
                    queue1.put(csi_i)

                   
                        

            s.close()

def process2(queue2):
    SIZE_BUFFER = 1440
    HZ = 800


    socket.NETLINK_CONNECTOR = 11
    CN_NETLINK_USERS = 11
    CN_IDX_IWLAGN = CN_NETLINK_USERS + 0xF
    NETLINK_ADD_MEMBERSHIP = 1

        
    address_src = ("10.20.14.28", 10086)
    address_des = ("10.20.14.37", 10010)
    # 创建socket对象，并绑定IP地址和端口号
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(address_des)
        while True:
            csi_L,address_src = s.recvfrom(SIZE_BUFFER)
            csi = np.frombuffer(csi_L,dtype=np.complex128).reshape((1,30,3,1))
            #csi = np.frombuffer(csi_L,dtype=np.complex128)
            queue2.put(csi)
            #print(csi[0,28,0,0])
            #csi_reshape = matrixreshape(csi)
            #print(csi_reshape[28,0])
        s.close()
            
def matrixreshape(original_array):
    length = original_array.shape[0]
    # 新数组形状为40x90
    reshaped_array = np.zeros((length, 90), dtype=np.complex128)

    # 将第三维的数据合并到新数组中
    for i in range(3):
        start_idx = i * 30
        end_idx = (i + 1) * 30
        reshaped_array[:, start_idx:end_idx] = original_array[:, :, i, 0]
    
    return reshaped_array.T

class RealTimePlot:
    def __init__(self):
        self.complex_data1 =complex(np.zeros(), np.zeros())  # 第一个复数数据数组
        self.complex_data2 = complex(np.zeros(), np.zeros())  # 第二个复数数据数组
        self.complex_data3 = complex(np.zeros(), np.zeros()) # 第三个复数数据数组

        self.fig, self.ax = plt.subplots(3, 1)

        # 设置初始范围
        self.ax[0].set_xlim(-1, 1)
        self.ax[0].set_ylim(-1, 1)
        self.ax[1].set_xlim(-1, 1)
        self.ax[1].set_ylim(-1, 1)
        self.ax[2].set_xlim(-1, 1)
        self.ax[2].set_ylim(-1, 1)

        # 设置初始标签
        self.ax[0].set_xlabel('Real')
        self.ax[0].set_ylabel('Imaginary')
        self.ax[0].set_title('Plot 1')
        self.ax[1].set_xlabel('Real')
        self.ax[1].set_ylabel('Imaginary')
        self.ax[1].set_title('Plot 2')
        self.ax[2].set_xlabel('Real')
        self.ax[2].set_ylabel('Imaginary')
        self.ax[2].set_title('Plot 3')

        plt.tight_layout()

        #self.animation = self.fig.canvas.new_timer(interval=100)
        #self.animation.add_callback(self.update_plot)
        #self.animation.start()

    def update_plot(self,loc,ne1,ne2):


        self.complex_data1=np.concatenate(np.concatenate,loc)  # 将新数据添加到第一个数组中
        self.complex_data2=np.concatenate(np.concatenate,ne1)  # 将新数据添加到第二个数组中
        self.complex_data2=np.concatenate(np.concatenate,ne2)  # 将新数据添加到第三个数组中

        # 绘制第一个图形
        self.ax[0].clear()
        self.ax[0].scatter(np.real(self.complex_data1), np.imag(self.complex_data1), c=np.arange(len(self.complex_data1)), cmap='cool', alpha=0.7)
        self.ax[0].set_xlim(-1, 5)
        self.ax[0].set_ylim(-1, 6)
        self.ax[0].set_xlabel('Real')
        self.ax[0].set_ylabel('Imaginary')
        self.ax[0].set_title('Plot 1')

        # 绘制第二个图形
        self.ax[1].clear()
        self.ax[1].scatter(np.real(self.complex_data2), np.imag(self.complex_data2), c='green', alpha=0.7)
        self.ax[1].set_xlim(-1, 1)
        self.ax[1].set_ylim(-1, 1)
        self.ax[1].set_xlabel('Real')
        self.ax[1].set_ylabel('Imaginary')
        self.ax[1].set_title('Plot 2')

        # 绘制第三个图形
        self.ax[2].clear()
        self.ax[2].scatter(np.real(self.complex_data3), np.imag(self.complex_data3), c='red', alpha=0.7)
        self.ax[2].set_xlim(-1, 1)
        self.ax[2].set_ylim(-1, 1)
        self.ax[2].set_xlabel('Real')
        self.ax[2].set_ylabel('Imaginary')
        self.ax[2].set_title('Plot 3')

        self.fig.canvas.draw()
if __name__ == '__main__':
    # 创建三个队列
    queue1 = multiprocessing.Queue()
    queue2 = multiprocessing.Queue()
    # 创建三个进程
    p1 = multiprocessing.Process(target=process1, args=(queue1,))
    p2 = multiprocessing.Process(target=process2, args=(queue2,))
    
    p1.start()
    p2.start()
       
    tx = 1 + 2j
    r1 = 0 + 0j

    # r2=3+0i
    r2 = 3 + 0j
    rx = [r1, r2]
    loc_ini = 0+1.5j
    
    
    plotWidget = RealTimePlot()
    
    while True:
        loc=[]
        csi_1=[]
        csi_2=[]
        while len (csi_1)<2400:
            data1 = queue1.get()
            csi_data1 = matrixreshape(data1)
            csi_1.append(csi_data1)
            data2 =  queue2.get()
            csi_data2 = matrixreshape(data2)
            csi_2.append(csi_data2)
        csi_d1=np.concatenate(csi_1,axis=1)
        csi_d2=np.concatenate(csi_2,axis=1)
        #print(csi_d1.shape)

        print(loc_ini)
        [loc,ne1,ne2] = process(csi_d1,csi_d2,800,loc_ini)
        
        plotWidget.update_plot(loc,ne1,ne2)
        
        loc_ini = loc[-1]
        plt.show()
        