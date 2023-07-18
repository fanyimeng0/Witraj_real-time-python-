import datetime
import os
import socket

import csiread
import numpy as np
import torch
from einops import rearrange
from rich.console import Console

from utils import setTorchDevice
from dataset import wt_filter

# ----------------------------------- Init ----------------------------------- #

console = Console()
device = setTorchDevice(-1)

# --------------------------------- constants -------------------------------- #


DTYPE_UINT8 = "uint8"
DTYPE_FLOAT32 = "float32"

SIZE_BUFFER = 4096
HZ = 256
TIME = 1000
NUM_TOTAL = HZ * TIME

socket.NETLINK_CONNECTOR = 11
CN_NETLINK_USERS = 11
CN_IDX_IWLAGN = CN_NETLINK_USERS + 0xF
NETLINK_ADD_MEMBERSHIP = 1


# ----------------------------------- args ----------------------------------- #


L = 256
FPS_RATIO = 4
FRAC = int(L / FPS_RATIO)
PATH = "./models/ml_shac5/0.pt"

Ntx = 1
Nrx = 3
Nsub = 30

csi_L = np.zeros((L, Nsub, Nrx, Ntx))
csi_frac = np.zeros((FRAC, Nsub, Nrx, Ntx))

model = torch.load(PATH, map_location=device)
model.eval()

# model_parameters = filter(lambda p: p.requires_grad, model.parameters())
# params = sum([np.prod(p.size()) for p in model_parameters])
# print("Model Parameters: " + str(params) + "\n")


# ----------------------------------- eval ----------------------------------- #


def label_map(idx):
    class_dict = {
        "bending": 0,
        "kicking": 1,
        "squating": 2,
        "standing": 3,
        "waving": 4,
    }
    return "{0} -> {1}".format(idx, list(class_dict.keys())[idx])


def format_batch(csi_L):
    # input (I, Nsub, Nrx, Ntx) -> output (B, C, H, W)
    # input (256, 30, 3, 1) -> output (1, 1, 90, 256)
    x = np.abs(rearrange(csi_L, "I N_s N_r N_t -> I N_t N_r N_s"))
    x = np.transpose(x.reshape(x.shape[0], -1))
    x = wt_filter(x, axis=1, wt="coif17", mode="garrote", level=2)
    x = np.array([[x]]).astype(DTYPE_FLOAT32)
    # print(x.shape)
    return torch.tensor(x)


def eval(X):
    # {'bending': 0, 'kicking': 1, 'squating': 2, 'standing': 3, 'waving': 4}
    pred = model(format_batch(X))
    # print(pred)
    pred_y = pred.argmax(1)
    # console.log(pred_y)
    console.log(label_map(pred_y.item()))


# --------------------------------- realtime --------------------------------- #


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
                        # print("one eval")
                        # print(csi_L.shape)
                        eval(csi_L)

        s.close()


# ----------------------------------- main ----------------------------------- #


if __name__ == "__main__":
    log_realtime()
