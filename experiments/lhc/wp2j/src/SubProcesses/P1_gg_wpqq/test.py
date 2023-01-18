import madevent
import random
import time
import sys 
import numpy as np
multi_channel_in = 1
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
nbatch = 10000
wgts = np.zeros(nbatch)
wgts_corr = np.zeros(nbatch)
#random.seed(9001)
chans = np.random.randint(3,5,size=nbatch)
for j in range(nbatch):
    R = np.array([random.random() for _ in range(20)])
    R[7] = 0.0
    #print(R)
    #print(f"Randon numbers: {R}")
    try:
        w = madevent.madevent_api(R, channel, True)
        alpha = madevent.get_multichannel()
        n_rand = madevent.get_number_of_random_used()
        r_ut = madevent.get_random_used_utility()
        wgts[j] = w
        wgts_corr[j] = np.nan_to_num(w/alpha[channel-1])
        #wgts[j] = w
        #print(n_rand, wgts[j], r_ut, alpha)
        #print(n_rand, wgts[j], r_ut)
        #print(f"bare weight: {wgts[j]}, correct weight: {wgts_corr[j]}, alpha: {alpha[channel-1]}")
        # print(f"alphas: {alpha}")
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
#print(wgts)
wgt2 = wgts**2
mean = np.mean(wgts)
error = np.sqrt((np.mean(wgt2) - mean**2)/(nbatch-1))
print(mean, error)

wgtc2 = wgts_corr**2
meanc = np.mean(wgts_corr)
errorc = np.sqrt((np.mean(wgtc2) - meanc**2)/(nbatch-1))
print(meanc, errorc)
