import multiprocessing
import time
from process import process
import os
import socket


import csiread
import numpy as np

import matplotlib.pyplot as plt
import time
from IPython.display import display, clear_output

    
def process1(queue1):
    # Save csi to queue 1
    SIZE_BUFFER = 3600
    HZ = 800
    TIME = 20000
    NUM_TOTAL = HZ * TIME

    socket.NETLINK_CONNECTOR = 11
    CN_NETLINK_USERS = 11
    CN_IDX_IWLAGN = CN_NETLINK_USERS + 0xF
    NETLINK_ADD_MEMBERSHIP = 1
    Ntx = 1
    Nrx = 3
    Nsub = 30

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
                    #csi for one sample

                   
                        

            s.close()

def process2(queue2):
    #save csi from 10.20.14.28 
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
    #change the csi shape to 90*time
    length = original_array.shape[0]
    reshaped_array = np.zeros((length, 90), dtype=np.complex128)

    # 将第三维的数据合并到新数组中
    for i in range(3):
        start_idx = i * 30
        end_idx = (i + 1) * 30
        reshaped_array[:, start_idx:end_idx] = original_array[:, :, i, 0]
    
    return reshaped_array.T

def plotall(input_array,input_array1,input_array2):
    fig = plt.figure(figsize=(8, 8))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[2, 1, 1])
    ax = fig.add_subplot(gs[0, 0])


    ax1 = fig.add_subplot(gs_bottom[0, 0])
    ax2 = fig.add_subplot(gs_bottom[0, 1])

    for i in range(100,len(input_array)):
        clear_output(wait=True)
        y = input_array[i-100:i]
        ax.plot(np.real(y),np.imag(y), color='r')
        ax.set_xlim(-1,4)
        ax.set_ylim(-1,7)

        ax.set_title('Trajectory')
        y1 = input_array1[i-100:i-50]
        y2 = input_array2[i-100:i-50]

        ax1.plot(np.real(y1),np.imag(y1), color='y')
        ax2.plot(np.real(y2),np.imag(y2), color='b')
        ax1.set_title('CSI Slope Rx1')
        ax2.set_title('CSI Slope Rx2')
        gs.tight_layout(fig)

        display(fig)

        display(fig)

        # 延时一段时间
        time.sleep(0.005)

        ax.clear()
        ax1.clear()
        ax2.clear()

       

if __name__ == '__main__':
    queue1 = multiprocessing.Queue()
    queue2 = multiprocessing.Queue()
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
    csi_1 = []
    csi_2 = []
    while True:
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
        csi_1=csi_1[-1000:]
        csi_2=csi_2[-1000:]

        print(loc_ini)
        [loc,ne1,ne2] = process(csi_d1,csi_d2,800,loc_ini)
        print(loc.shape)
        loc_ini = loc[-1]
        
        plotall(loc,ne1,ne2)
        
        
    
        plt.grid(True)


        
