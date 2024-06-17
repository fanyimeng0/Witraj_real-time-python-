import socket
import csiread
import numpy as np
import pickle as pk
import os
# --------------------------------- constants -------------------------------- #

SIZE_BUFFER = 3600
NUM_TOTAL = 2000000 * 1000  # Assuming HZ = 1000 and TIME = 2000

socket.NETLINK_CONNECTOR = 11
CN_NETLINK_USERS = 11
CN_IDX_IWLAGN = CN_NETLINK_USERS + 0xF
NETLINK_ADD_MEMBERSHIP = 1

# --------------------------------- realtime --------------------------------- #

Ntx = 1
Nrx = 3
Nsub = 30

def log_realtime():
    address_src = ('192.168.50.51', 9995)
    target_ip = '192.168.50.250'  # Target IP address
    target_port = 10010  # Target port

    csidata = csiread.Intel(None, Nrx, Ntx)
    count = 0

    with socket.socket(socket.AF_NETLINK, socket.SOCK_DGRAM, socket.NETLINK_CONNECTOR) as s, \
            socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_socket:
        s.bind((os.getpid(), CN_IDX_IWLAGN))
        s.setsockopt(270, NETLINK_ADD_MEMBERSHIP, CN_IDX_IWLAGN)
        udp_socket.bind(address_src)
        while True:
            ret = s.recv(SIZE_BUFFER)
            cnmsg_data = ret[36:]  # Keep only data part
            status = csidata.pmsg(cnmsg_data)
            if status == 0xBB:  # If status is valid
                csi_i = csidata.get_scaled_csi()
                print(csi_i[:,10,1,0])
                send_data_to_ip(csi_i, target_ip, target_port)
                #count += 1
  # Ensure to increment the count

                #    print(result)
def send_data_to_ip(data, target_ip, target_port):
    serialized_data = pk.dumps(data)  # 序列化数据
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # 创建 UDP socket
   
    sock.sendto(serialized_data, (target_ip, target_port))  # 发送数据
    #print(f"数据已发送到 {target_ip}:{target_port}")
    
    sock.close()  # 关闭 socket   
# ----------------------------------- main ----------------------------------- #

if __name__ == "__main__":
    log_realtime()
