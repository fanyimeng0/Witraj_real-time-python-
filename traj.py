import threading
import socket
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
from process import process

cache_len = 1000
shape = (1,30, 3, 1)
complex_zero_array = np.zeros(shape, dtype=complex)
cache_len = 2000
cache_csi1 = [np.zeros(shape, dtype=complex) for _ in range(cache_len)]
cache_csi2 = [np.zeros(shape, dtype=complex) for _ in range(cache_len)]
tx = 3.2 + 0j
r1 = 0 - 0j
r2 = 3.2 + 2.7j
rx = [r1, r2]
loc_ini_x = 1.8
loc_ini_y = 1.35
cache_traj_x = [loc_ini_x]*1000
cache_traj_y = [loc_ini_y]*1000


mutex = threading.Lock()
def matrixreshape(original_array):
    #change the csi shape to 90*time
    length = original_array.shape[0]
    reshaped_array = np.zeros((length, 90), dtype=np.complex128)
    for i in range(3):
        start_idx = i * 30
        end_idx = (i + 1) * 30
        reshaped_array[:, start_idx:end_idx] = original_array[ :,:, i]
    
    return reshaped_array.T
class GetDataThread(threading.Thread):
    def __init__(self, device, ip='0.0.0.0', port1=10010,port2=10020):
        super(GetDataThread, self).__init__()
        self.ip = ip
        self.port1 = port1
        self.port2 = port2
        # 如果需要根据设备类型初始化不同的设置
        self.device = device

    def run(self):
        thread1 = threading.Thread(target=self.rx1)
        thread2 = threading.Thread(target=self.rx2)
        thread1.start()
        thread2.start()


    def rx1(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((self.ip, self.port1))  # 绑定到指定的IP和端口

            while True:
                count = 0
                data, addr = sock.recvfrom(4096)  # 接收数据
                csi = pickle.loads(data)  # 反序列化数据
                mutex.acquire()
                cache_csi1.pop(0)
                cache_csi1.append(csi)
                mutex.release()

                # 假设receive_data是一个numpy数组，这里打印特定的数据点
                #print(receive_data[:, 10, 1, 0])
    
    def rx2(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((self.ip, self.port2))  # 绑定到指定的IP和端口
            count = 0
            while True:
                
                data, addr = sock.recvfrom(4096)  # 接收数据
                csi = pickle.loads(data)  # 反序列化数据
                msg_len = len(csi)
                mutex.acquire()
                cache_csi2.pop(0)
                cache_csi2.append(csi)
                mutex.release()
                count += 1   
                if count % 2000 == 0:
                    loc_ini = cache_traj_x[-1]+cache_traj_y[-1]*1j
                    mutex.acquire()
                    #print(cache_csi1[0].shape)
                    merged_csi1=np.squeeze(np.stack(cache_csi1,axis=0))
                    merged_csi2=np.squeeze(np.stack(cache_csi2,axis=0))
                
                    csi_1 = matrixreshape(merged_csi1)
                    csi_2 = matrixreshape(merged_csi2)
                    loc = process(csi_1,csi_2,1000,loc_ini,tx,rx)
                    print(loc[-1])
                    length = len(loc)
                    for i in range (0,length):
                        cache_traj_x.pop(0)
                        cache_traj_x.append(loc[i].real)
                        cache_traj_y.pop(0)
                        cache_traj_y.append(loc[i].imag)
                    mutex.release()  


# 动态绘制(x, y)点的函数
def realtime_traj():
    fig, ax = plt.subplots()
    plt.title('Wifi Trajectory')
    plt.xlabel('X(m)')
    plt.ylabel('Y(m)')
    line1, = ax.plot([], [])

    tx_x = [3.2]
    tx_y = [0]
    rx_x = [0, 3.2]
    rx_y = [0, 2.7]
    init_x = [1.5]
    init_y = [1.2]
    ax.plot(tx_x, tx_y, 'x')
    ax.plot(rx_x, rx_y, 'o')
    ax.plot(init_x, init_y, 'x', color='r')
    ax.set_aspect('equal') 
    

    def init():
        ax.set_xlim(-3,5)
        ax.set_ylim(-2,8)
        return line1,

    def animate(i):
        global cache_traj_x,cache_traj_y,mutex
        mutex.acquire()
        line1.set_data(cache_traj_x,cache_traj_y)
        mutex.release()
        return line1,
    
    ani =FuncAnimation(fig, animate, init_func=init, interval=1000/25, blit=True)
    plt.show()


def realtime_plot(device):
    task1 = GetDataThread(device)
    task1.start()
    realtime_traj()
if __name__ == '__main__':
    realtime_plot('intel')