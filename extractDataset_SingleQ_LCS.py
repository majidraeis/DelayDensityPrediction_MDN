import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import matplotlib.pyplot as plt
import copy
import csv
from tqdm import tqdm
import random

class Job:
    def __init__(self, Ta,Ts,B0,index,serviceHistLength, delayHistLength):
        # self.Ta = Ta
        # self.Ts = Ts
        # self.Td = Td
        # self.Tw = Tw
        # self.Ba = Ba #number of customors in the system (including servers) upon arrival
        # self.index = index
        # self.b = B0
        # self.serviceHistLength= serviceHistLength
        Td= np.zeros(length)
        Tw= np.zeros(length)
        Ba= np.zeros(length)
        MA= np.zeros(length)
        if serviceHistLength==0 and delayHistLength==0:
            self.dict = dict(zip(index , list(map(list,list(zip(Ta,Td,Ts,Tw,Ba,MA))))))
        elif serviceHistLength==0 or delayHistLength==0:
            service_delay_History= np.zeros((np.shape(Ta)[0],max(serviceHistLength,delayHistLength)))
            self.dict = dict(zip(index , np.append(list(map(list,list(zip(Ta,Td,Ts,Tw,Ba,MA)))), service_delay_History, axis=1) ))
        else:
            serviceHistory= np.zeros((np.shape(Ta)[0],serviceHistLength))
            delayHistory= np.zeros((np.shape(Ta)[0],delayHistLength))
            self.dict= dict(zip(index , np.append(np.append(list(map(list,list(zip(Ta,Td,Ts,Tw,Ba,MA)))), serviceHistory, axis=1),delayHistory,axis=1) ))
        # Dictionary values:  1-ArrivalTime 2-DepartureTime 3-ServiceTime 4-WaitingTime 5-BackloggUponArrival 6-ServiceTimeMovingAverage
        # 7-...-ServiceTimes of (serviceHistLength) previous jobs [optional] 8-...-Delays of (delayHistLength) previous jobs [optional]

def infQueueMultiServ_CHANjob(job,MAlength,N0,b0,serviceHistLength,delayHistLength,QLnoise="False",QLnoise_sigma=0):
    jobMatrix= np.array(list(job.dict.values()))
    jobIndex= np.array(list(job.dict.keys()))
    length= len(jobIndex)
    order=np.argsort(jobMatrix[:, 0])
    Ta= jobMatrix[:,0][order]
    Ts= jobMatrix[:,2][order]
    # mean_Ts= np.mean(Ts)
    jobIndexOrdered= jobIndex[order]
    indexout= []
    f= Ta[0:N0]+Ts[0:N0]
    d= sorted(f)
    index= jobIndexOrdered[np.argsort(f)]
    backlogD=[[0,0,0]]
    backlogA=[]
    jobDepCum=[]
    for i in tqdm(range(0, length)):
        jobDep, d= d[0], np.delete(d,0)   
        jobDepCum.append(jobDep)
        indexout= np.append(indexout , index[0])
        index= np.delete(index , 0)  
        tempOcc= np.sum(Ta <= jobDep)-(i+1)
        backlogD.append([jobDep , max(tempOcc - N0, 0) , tempOcc])
        tempOcc= i+1 - np.sum( jobDepCum <= Ta[i])
        backlogA.append( [Ta[i] , max(tempOcc - N0, 0) , tempOcc] )
        job.dict[jobIndex[i]][4] =  np.round(max( i - np.sum( jobDepCum <= Ta[i]) + (QLnoise=="True")*np.random.normal(0,QLnoise_sigma) , 0)) # num of customors in the system (including servers) right upon i'th arrival
        
        if i <= length-N0-1:
            F= max(Ta[N0+i] , jobDep)+Ts[N0+i]
            u= np.append(F , d)
            u= np.maximum(u , F)
            addedIndex=np.sum(u==F)-1
            d= np.minimum(np.append(d,np.inf),u)
            index= np.append(index[:addedIndex], np.append(jobIndexOrdered[N0+i] , index[addedIndex:]))
        job.dict[indexout[-1]][1]= jobDep
        # job.dict[indexout[-1]][3]= max(0 , jobDep - job.dict[indexout[-1]][0]) # delay (Sojourn)
        job.dict[indexout[-1]][3]= max(0 , jobDep - job.dict[indexout[-1]][0] - job.dict[indexout[-1]][2]) #waiting
        
        if np.sum(jobDepCum < Ta[i]) > MAlength-1:
            jobDepSorted= np.sort(jobDepCum)
            indexx=indexout[np.argsort(jobDepCum)]
            if serviceHistLength > 0: #creating service history
                job.dict[jobIndex[i]][-(delayHistLength+serviceHistLength):-delayHistLength] =  [job.dict[j][2] for j in indexx[jobDepSorted < Ta[i]][-(serviceHistLength+delayHistLength):-delayHistLength]]
            if delayHistLength > 0: #creating delay history
                job.dict[jobIndex[i]][-delayHistLength:] =  [job.dict[j][3] for j in indexx[jobDepSorted < Ta[i]][-delayHistLength:]]
            job.dict[jobIndex[i]][5]= np.mean([job.dict[j][2] for j in indexx[jobDepSorted < Ta[i]][-MAlength:]])
        
    jobMatrix= np.array(list(job.dict.values()))        
    # backlogA= [(Ta[i] , max(i+1 - np.sum(np.sort(jobMatrix[:,1]) <= Ta[i]) - N0, 0)) for i in range(0, length)]    
    backlogA= np.concatenate((backlogA,backlogD))
    backlogA= np.array(sorted(backlogA, key=lambda x: x[0]))
    job.b= backlogA
    job.index= indexout.astype(int)
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
def PassengerArrival(mean_Ta=0.8, mean_Tf=100, flightNum= 100):
    # units of mean_Tf and mean_Ta are minutes
    # flightNum= 100
    delay = 0 #set to one for having delay
    delayMean = 10
    delayStd = 2
    passengerNumStd = 10
    passengerNumMean = 50
    passengerNumMax = 60
    flightInterArr = np.random.exponential(mean_Tf, flightNum)
    flightArrivals = np.cumsum(flightInterArr)
    passengerArrivals = []
    passengerFlightNo = []
    for i in range(flightNum):
        passengerNum= int(max(passengerNumMax, np.floor(np.random.normal(passengerNumMean,passengerNumStd))))
        interArr_Passenger= np.random.exponential(mean_Ta , passengerNum)
        rel_PassengerArrival= (delay==1)*np.random.normal(delayMean,delayStd) + np.cumsum(interArr_Passenger)
        passengerArrivals= np.append(passengerArrivals, rel_PassengerArrival + flightArrivals[i])
        passengerFlightNo= np.append(passengerFlightNo, i*np.ones(passengerNum))
    
    return np.sort(passengerArrivals), flightArrivals
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
def MarkovOnOff(length, pOnOff, pOffOn, mean_Ta_state):
    s = 0
    interArr = np.zeros(length)
    # for i in range(length):
    i = 0
    while i < length:
        while s and i < length:
            interArr[i] = np.random.exponential(mean_Ta_state)
            s = np.random.uniform() > pOnOff
            i += 1
        idle_time = 0
        while not s and i < length:
            idle_time += np.random.exponential(mean_Ta_state)
            # interArr[i] = np.random.exponential(mean_Ta_state)
            s = np.random.uniform() < pOffOn
            if s:
                interArr[i] = idle_time
                i += 1
    Ta = np.cumsum(interArr)
    return Ta
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\INITIALIZATION\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
training = 0 # set to one(zero) for generating training(test) data
ArrivalType = "exp" #Passenger or exp
ServiceType = "exp" #exp or gauss
seed = 3
lengthTa, FN = 100000, 200
train_length = 100000
test_length = 2000
rho = 0.99
plot_flag = 1
MAlength, N, b0, serviceHistLength, delayHistLength = 100, 1, 0, 0, 1
mean_Ts = 1
mean_Ta = mean_Ts/(N*rho)
if training == 0:
    seed = 20
np.random.seed(seed)
QLnoise, QLnoise_sigma = "False", 5
ServiceRangeLen = 1
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

if training:
    # ////////////////arrival  times  
    Ts = []
    if ArrivalType == "exp":
        # mean_Ta= 1*np.arange(1,ServiceRangeLen+1)
        interArr = []
        for i in range(ServiceRangeLen):
            interArr = np.append(interArr, np.random.exponential(mean_Ta, lengthTa))
            # Ts=np.append(Ts, np.random.exponential(mean_Ts[i] , lengthTa)) #exponential 
            # Ts=np.append(Ts, np.random.normal(mean_Ts[i] ,sigma_s[i], lengthTa)) #Gaussian
        Ta = np.cumsum(interArr)

    # elif ArrivalType == "Passenger":
    #     mean_Ta= 0.4*np.arange(1, ServiceRangeLen+1)
    #     Ta=[]
    #     for i in range(ServiceRangeLen):
    #         if i==0:
    #             Ta_shift= 0
    #         else:
    #             Ta_shift= Ta[-1]
    #         Ta_temp,Tf= PassengerArrival(mean_Ta[i], mean_Tf=100,flightNum=FN)
    #         Ta= np.append(Ta, Ta_temp+Ta_shift)
    #         # Ts=np.append(Ts, np.random.normal(mean_Ts[i] , sigma_s[i], len(Ta_temp))) #Gaussian
    #     mean_Ts= N*np.mean(np.diff(Ta))/1.01
            
    
    length= len(Ta)
    if ServiceType=="exp": #exponential 
        Ts= np.random.exponential(mean_Ts, length) 
        
    elif ServiceType=="gauss": #for Gaussian service
        sigma_s= 0.33* mean_Ts
        Ts= np.random.normal(mean_Ts , sigma_s, length).clip(min=0)+np.exp(-10)
        
else:
    
    # ////////////////arrival  times  
    if ArrivalType == "exp":
        # mean_Ts= 10 
        # sigma_s= 1*mean_Ts#for Gaussian service
        interArr = np.random.exponential(mean_Ta, lengthTa)
        # interArr= mean_Ta *np.ones(lengthTa)
        Ta= np.cumsum(interArr)
        # mean_Ts= N*np.mean(np.diff(Ta))/1.01

    # elif ArrivalType == "Passenger":
    #     mean_Ta= 0.4
    #     Ta, Tf= PassengerArrival(mean_Ta, mean_Tf=100,flightNum=FN)
    #     mean_Ts= N*np.mean(np.diff(Ta))/1.01
    #     # mean_Ts= N*mean_Ta/1.1
    #     # sigma_s= 1* mean_Ts#for Gaussian service
    # ///////////////////service times
    # Ts= np.random.exponential(mean_Ts , len(Ta)) #exponential  
    # Ts= np.random.normal(mean_Ts, sigma_s, len(Ta)) #Gaussian  
    
    length = len(Ta)
    if ServiceType=="exp":     
        Ts = np.random.exponential(mean_Ts, length) #exponential
        
    elif ServiceType=="gauss": #for Gaussian service
        sigma_s= 0.33* mean_Ts
        Ts= np.random.normal(mean_Ts, sigma_s,length).clip(min=0)+np.exp(-10)
    

# job= Job(Ta, Ts, np.zeros(length), np.zeros(length), np.zeros(length) ,B0 , np.arange(1,length+1),serviceHistLength)
job = Job(Ta, Ts, b0, np.arange(1, length+1), serviceHistLength, delayHistLength)
infQueueMultiServ_CHANjob(job, MAlength, N, b0, serviceHistLength, delayHistLength, QLnoise, QLnoise_sigma)
jobMatrix = np.array(list(job.dict.values()))
jobIndex = np.array(list(job.dict.keys()))
    
order = np.argsort(jobMatrix[:, 0])
Ta = np.sort(jobMatrix[:, 0])
Td = np.sort(jobMatrix[:, 1])

if plot_flag:

    plt.figure()
    ax1 = plt.subplot(311)
    plt.step(*zip(*job.b[:, :2]), where='post')
    plt.ylabel('Backlog')
    plt.xlabel('Time')
    plt.title('Number of servers ($N_{%s}$)= %s'%(1,N))
    plt.show()

    plt.subplot(312,sharex=ax1)
    line_up, = plt.step(np.append(np.append(0,Ta), Td[-1]),np.append(range(0,length+1),length),'g',where='post', label='Cumalative arrival')
    line_down, = plt.step(np.append(0,Td),range(0,length+1),'r',where='post',label='Cumulative departure')
    plt.legend(handles=[line_up, line_down])
    plt.ylabel('Number of passengers')
    plt.xlabel('Time (mins)')
    plt.show()

    if ArrivalType == "Passenger":
        plt.subplot(313,sharex=ax1)
        plt.scatter(Tf, np.ones(np.size(Tf)))
        plt.xlabel('Time (mins)')
        plt.ylabel('Flight Arrival')
        plt.ylim(0.5,1.5)
        plt.show()



# ///////////////////Data set for delay estimation//////////////
for k in np.arange(length):
    if jobMatrix[k,-1]!=0:
        begin= k
        break
        
if serviceHistLength > 0:
    serviceHistory= jobMatrix[begin:,-(serviceHistLength+delayHistLength):-delayHistLength]
if delayHistLength > 0:
    delayHistory= jobMatrix[begin:,-delayHistLength:]
#     serviceHistory=[]
occupancy= jobMatrix[:,4].astype(int)
occ= occupancy - N
backlog= occ.clip(min=0)
freeServers= -occ.clip(max=0)
arrival= jobMatrix[:,0] #for plotting delay vs arrival times
# departure= jobMatrix[:,1] #for plotting 
delay= jobMatrix[:,3] 
MA= jobMatrix[:,5]

# Data Structure: 1-ServiceHistory(optional) 2-backlogUponArrival 3-#ofFreeServersUponArrival 4-ServiceTimeMovingAverage 5-SojournTime

if training:
# \\\\\\\\\\\\\\\\\\\train data \\\\\\\\\\\\\\\
    # data= np.array(list(zip(backlog[begin:], freeServers[begin:], MA[begin:])))
    data = np.array(list(zip(freeServers[begin:], MA[begin:]))) ## without queuelength information 
    if serviceHistLength > 0:
        data = np.append(data, serviceHistory, axis=1)
    if delayHistLength > 0:
        data = np.append(data, delayHistory, axis=1)
        
    data = np.append(data, delay[begin:].reshape(-1, 1), axis=1)
    # Shuffle the Dataset set
    totSamples = np.shape(data)[0]
    order = np.argsort(np.random.random(totSamples))[:train_length]
    data = data[order]
    with open('DelayPredTrainingdata.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)
    
    csvFile.close()
else:
    # \\\\\\\\\\\\\\\\test data\\\\\\\\\\\\\\\

    data = np.array(list(zip(backlog[begin:], freeServers[begin:], MA[begin:])))
    if serviceHistLength > 0:
        data = np.append(data, serviceHistory, axis=1)
    if delayHistLength > 0:
        data = np.append(data, delayHistory, axis=1)
    # data = np.append(departure[begin:].reshape(-1,1) ,data, axis=1)
    data = np.append(arrival[begin:].reshape(-1, 1), data, axis=1)
    data = np.append(data, delay[begin:].reshape(-1, 1), axis=1)
    # 1)arrival 2)backlog 3)#ofFreeServers 4)serviceTimeMovingAverage 5)service hist 6)delay hist 7)label

    # Shuffle the Dataset set
    totSamples = np.shape(data)[0]
    order = np.argsort(np.random.random(totSamples))[:test_length]
    data = data[order]

    with open('DelayPredTestdata.csv', 'w') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerows(data)

    csvFile.close()

if plot_flag:
    plt.figure()
    # plt.hist(data[:, -1], 50, alpha=0.5, label='GroundTruth')
    # plt.hist(data[:, -2], 50, alpha=0.5, label='LCS')
    plt.hist([data[:, -1], data[:, -2]], 50, label=['GroundTruth', 'LCS'])
    plt.xlabel('Delay')
    plt.legend(loc='upper right')

    if training:
        plt.title('Training')
    else:
        plt.title('Test')
    plt.show()

