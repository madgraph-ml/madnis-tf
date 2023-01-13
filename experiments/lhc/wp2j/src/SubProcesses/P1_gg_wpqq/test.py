import madevent
import random
import time
import sys 
import numpy as np
multi_channel_in = 0
helicity_sum = 1
channel = 1
dconfig = channel


madevent.configure_code(multi_channel_in,helicity_sum,dconfig)
# R= [random.random() for _ in range(20)]
# wgt = madevent.madevent_api(R,1,True)
# p = madevent.get_momenta()
# print(p.T)
# print(wgt)
# print("nran = ", madevent.get_number_of_random_used())

i=0
wgt =0
start = time.time()
last =  1
work = 0
index= []
index_fail = []
nbatch = 100
wgts = np.zeros(nbatch)
chans = np.random.randint(3,5,size=nbatch)
chans[0] = 8
for j in range(nbatch):
    R = np.array([random.random() for _ in range(20)])
    try:
        wgts[j] = madevent.madevent_api(R, chans[0], True)
        n_rand = madevent.get_number_of_random_used()
        r_ut = madevent.get_random_used_utility()
        print(n_rand, wgts[j], r_ut)
    except:
        pass
#        p = madevent.get_momenta()
#        print(p.T)
#        print("value is ", wgt)
#        i+=1
#        current = time.time() - start
#        if wgt == 0:
#            last = current +1
#            print(i, current)
#            if len(index_fail) > 100:
#            index_fail.append(j)
#                break
#            else:
#                index_fail.append(j)
#        else:
#            work+=1
#            index.append(madevent.get_number_of_random_used())
    # print(i, current)

#print(chans[0])
n_rand = madevent.get_number_of_random_used()
print(n_rand)
wgt2 = wgts**2
mean = np.mean(wgts)
error = np.sqrt((np.mean(wgt2) - mean**2)/(nbatch-1))
print(mean, error)
