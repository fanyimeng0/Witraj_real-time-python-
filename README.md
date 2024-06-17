# Witraj_real-time_python

 This is a repo for python-based real-time Wi-Fi tracking by using Intel 5300. The algorithm is forked from https://github.com/Soccerene/WiTraj

## How to Run this code

Before using this code, make sure you have configured the driver correctly. 

Then you need to install csiread https://github.com/citysu/csiread, for python>=3.8 

```
pip install csiread
```

Also you need to change the source code of from scipy.signal import savgol_filter like this since we use complex numbers.

```python
    x = np.asarray(x)
    # Ensure that x is either single or double precision floating point.
    #  if x.dtype != np.float64 and x.dtype != np.float32:
    #    x = x.astype(np.float64)
```

For the Tx, just send packets as CSI Tool defines. For Rx1 and Rx2, run send_to_pc.py, you need to change your corresponding IP. For the PC real-time showing the result, run traj.py, make sure traj.py and process.py in the folder. Also change the corresponding location of TX and RX.

If you want to verify the tracking algorithm's performance, run process.py, you can change the csifile by changing the files , here we use our data while people walking as a square.

The real_time tracking performance like this:

 ![Real-time tracking ](wifisensing.gif)

