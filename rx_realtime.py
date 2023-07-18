import datetime
import os
import socket

import csiread
import numpy as np

# --------------------------------- constants -------------------------------- #


SIZE_BUFFER = 4096
HZ = 256
TIME = 2
NUM_TOTAL = HZ * TIME

socket.NETLINK_CONNECTOR = 11
CN_NETLINK_USERS = 11
CN_IDX_IWLAGN = CN_NETLINK_USERS + 0xF
NETLINK_ADD_MEMBERSHIP = 1


# --------------------------------- realtime --------------------------------- #


L = 256  # fixed window length
FPS_RATIO = 8  # eval per second
FRAC = int(L / FPS_RATIO)

Ntx = 1
Nrx = 3
Nsub = 30

csi_L = np.zeros((L, Nsub, Nrx, Ntx))
csi_frac = np.zeros((FRAC, Nsub, Nrx, Ntx))


def log_realtime():
    global csi_L, csi_frac

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

                # merge csi_i into csi_frac
                csi_frac = np.concatenate((csi_frac[1:], csi_i), axis=0)
                # print(101, csi_frac.shape) # (32, 30, 3, 1)

                # print count
                if count == 1 or count % HZ == 0:
                    # print(csi_i.shape) # (1, 30, 3, 1)
                    print("count: %d\ttime: %s" % (count, datetime.datetime.now()))

                # merge csi_frac into csi_L
                if count % FRAC == 0:
                    csi_L = np.concatenate((csi_L[FRAC:], csi_frac), axis=0)
                    # print(102, csi_L.shape) # (256, 30, 3, 1)
                    if count >= L:
                        # send csi_L as one input
                        print("one eval")
                        #print(csi_L.shape)

        s.close()


# ----------------------------------- main ----------------------------------- #


if __name__ == "__main__":
    log_realtime()
