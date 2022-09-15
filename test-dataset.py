from itertools import count
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def create_nab_nyc_dataset():
    train_nab=np.load("processed/NAB/nyc_taxi_train.npy")
    test_nab=np.load("processed/NAB/nyc_taxi_test.npy")
    label_nab=np.load("processed/NAB/nyc_taxi_labels.npy")

    # print(train_nab.shape,test_nab.shape,label_nab.shape)
    # print(train_nab[:5],test_nab[:5],label_nab[:5])
    # print(np.sum(label_nab))

    assert np.allclose(train_nab,test_nab)==True, "OK SONO UGUALI"

    dataset_nab_nyc=pd.read_csv("data/NAB/nyc_taxi.csv")
    dataset_nab_nyc['anomaly']=0

    for timestamp_anomaly in ["2014-11-01 19:00:00","2014-11-27 15:30:00","2014-12-25 15:00:00","2015-01-01 01:00:00","2015-01-27 00:00:00"]:
        print(dataset_nab_nyc[dataset_nab_nyc['timestamp']==timestamp_anomaly]['anomaly'])

    # print(dataset_nab_nyc[dataset_nab_nyc['timestamp']=="2014-11-01 19:00:00"]['anomaly'])
    dataset_nab_nyc.loc[5942,'anomaly']=1
    dataset_nab_nyc.loc[7183,'anomaly']=1
    dataset_nab_nyc.loc[8526,'anomaly']=1
    dataset_nab_nyc.loc[8834,'anomaly']=1
    dataset_nab_nyc.loc[10080,'anomaly']=1

    print(dataset_nab_nyc['anomaly'].sum())
    dataset_nab_nyc.to_csv("Dataset_AD/nab_nyc.csv")

def create_nab_machine_failure_dataset():
    
    dataset_nab_machine_failure=pd.read_csv("data/NAB/machine_temperature_system_failure.csv")
    dataset_nab_machine_failure['anomaly']=0

    for timestamp_anomaly in [ "2013-12-11 06:00:00", "2013-12-16 17:25:00", "2014-01-28 13:55:00","2014-02-08 14:30:00"]:
        print("TA",dataset_nab_machine_failure[dataset_nab_machine_failure['timestamp']==timestamp_anomaly]['anomaly'])

    # print(dataset_nab_nyc[dataset_nab_nyc['timestamp']=="2014-11-01 19:00:00"]['anomaly'])
    dataset_nab_machine_failure.loc[2409,'anomaly']=1
    dataset_nab_machine_failure.loc[3986,'anomaly']=1
    dataset_nab_machine_failure.loc[16340,'anomaly']=1
    dataset_nab_machine_failure.loc[19515,'anomaly']=1
    # dataset_nab_nyc.loc[10080,'anomaly']=1
    print(dataset_nab_machine_failure)
    print(dataset_nab_machine_failure['anomaly'].sum())
    # print(dataset_nab_machine_failure.head(5))
    dataset_nab_machine_failure.to_csv("Dataset_AD/nab_machine_failure.csv")

def create_nab_ec2_request_latency_system_failure():
    dataset_nab_machine_failure=pd.read_csv("data/NAB/ec2_request_latency_system_failure.csv")
    dataset_nab_machine_failure['anomaly']=0

    for timestamp_anomaly in [  "2014-03-14 09:06:00","2014-03-18 22:41:00", "2014-03-21 03:01:00"]:
        print("TA",dataset_nab_machine_failure[dataset_nab_machine_failure['timestamp']==timestamp_anomaly]['anomaly'])

    # print(dataset_nab_nyc[dataset_nab_nyc['timestamp']=="2014-11-01 19:00:00"]['anomaly'])
    dataset_nab_machine_failure.loc[2081,'anomaly']=1
    dataset_nab_machine_failure.loc[3395,'anomaly']=1
    dataset_nab_machine_failure.loc[4023,'anomaly']=1

    print(dataset_nab_machine_failure)
    print(dataset_nab_machine_failure['anomaly'].sum())
    # print(dataset_nab_machine_failure.head(5))
    dataset_nab_machine_failure.to_csv("Dataset_AD/nab_ec2_request_latency.csv")

def create_nab_cpu_utilization():
    dataset_nab_machine_failure=pd.read_csv("data/NAB/cpu_utilization_asg_misconfiguration.csv")
    dataset_nab_machine_failure['anomaly']=0

    for timestamp_anomaly in ["2014-07-12 02:04:00", "2014-07-14 21:44:00"]:
        print("TA",dataset_nab_machine_failure[dataset_nab_machine_failure['timestamp']==timestamp_anomaly]['anomaly'])

    # print(dataset_nab_nyc[dataset_nab_nyc['timestamp']=="2014-11-01 19:00:00"]['anomaly'])
    dataset_nab_machine_failure.loc[17002,'anomaly']=1
    dataset_nab_machine_failure.loc[17814,'anomaly']=1


    print(dataset_nab_machine_failure)
    print(dataset_nab_machine_failure['anomaly'].sum())

    print(dataset_nab_machine_failure.head(5))
    dataset_nab_machine_failure.to_csv("Dataset_AD/nab_cpu_utilization.csv")

def create_nab_ambient_temperature():
    dataset_nab_machine_failure=pd.read_csv("data/NAB/ambient_temperature_system_failure.csv")
    dataset_nab_machine_failure['anomaly']=0

    for timestamp_anomaly in [  "2013-12-22 20:00:00", "2014-04-13 09:00:00"]:
        print("TA",dataset_nab_machine_failure[dataset_nab_machine_failure['timestamp']==timestamp_anomaly]['anomaly'])

    # print(dataset_nab_nyc[dataset_nab_nyc['timestamp']=="2014-11-01 19:00:00"]['anomaly'])
    dataset_nab_machine_failure.loc[3721,'anomaly']=1
    dataset_nab_machine_failure.loc[6180,'anomaly']=1

    print(dataset_nab_machine_failure)
    print(dataset_nab_machine_failure['anomaly'].sum())
    print(dataset_nab_machine_failure.head(5))
    dataset_nab_machine_failure.to_csv("Dataset_AD/nab_ambient_temperature.csv")

def create_nab_rogue_updown():
    dataset=pd.read_csv("data/NAB/rogue_agent_key_updown.csv")
    dataset['anomaly']=0

    for timestamp_anomaly in [ "2014-07-15 04:00:00", "2014-07-17 08:50:00"]:
        print("TA",dataset[dataset['timestamp']==timestamp_anomaly]['anomaly'])

    # print(dataset_nab_nyc[dataset_nab_nyc['timestamp']=="2014-11-01 19:00:00"]['anomaly'])
    dataset.loc[2375,'anomaly']=1
    dataset.loc[3009,'anomaly']=1

    
    print(dataset)
    print(dataset['anomaly'].sum())

    print(dataset.head(5))
    dataset.to_csv("Dataset_AD/nab_rogue_updown.csv")

def create_nab_rogue_hold():
    dataset=pd.read_csv("data/NAB/rogue_agent_key_hold.csv")
    dataset['anomaly']=0

    for timestamp_anomaly in [  "2014-07-15 08:30:00",  "2014-07-17 09:50:00"]:
        print("TA",dataset[dataset['timestamp']==timestamp_anomaly]['anomaly'])

    # print(dataset_nab_nyc[dataset_nab_nyc['timestamp']=="2014-11-01 19:00:00"]['anomaly'])
    dataset.loc[716,'anomaly']=1
    dataset.loc[1287,'anomaly']=1
    
    print(dataset)
    print(dataset['anomaly'].sum())

    print(dataset.head(5))
    dataset.to_csv("Dataset_AD/nab_rogue_hold.csv")

          

# create_nab_nyc_dataset()
# create_nab_machine_failure_dataset()
# create_nab_ec2_request_latency_system_failure()
# create_nab_cpu_utilization()
# create_nab_ambient_temperature()
# create_nab_rogue_updown()
# create_nab_rogue_hold()

## MBA

def create_mba_dataset():

    # dataset_train=pd.read_excel("data/MBA/train.xlsx")
    
    # dataset_train.to_csv("Dataset_AD/train.csv")
    dataset_final=pd.read_csv("Dataset_AD/train.csv")
    dataset_final_label=np.loadtxt("Dataset_AD/labels_mba.txt")
    print(dataset_final_label.shape)
    dataset_final=dataset_final.drop(0,axis=0)
    dataset_final['anomaly']=dataset_final_label[:,0]
    dataset_final=dataset_final.drop(['Unnamed: 0','sample'],axis=1)
    print(dataset_final.head(5))
    print(dataset_final.shape)
    print(dataset_final['anomaly'].sum())
    print((dataset_final[dataset_final['anomaly']==1]))
    # print(dataset_final['sample'].nunique())
    # # print(dataset_final,dataset_final_label)
    # print(dataset_final_label[:,0].sum())
    # print(dataset_final_label[:,1].sum())
    # print(np.where(dataset_final_label[:,0]==1))
    
    # print(np.allclose(dataset_final_label[:,0],dataset_final_label[:,1]))
    
    dataset_final.to_csv("Dataset_AD/mba_dataset.csv")



# create_mba_dataset()


#SMD

def create_smd_dataset():

    X_train=np.load("processed/SMD/machine-1-6_train.npy")
    X_test=np.load("processed/SMD/machine-1-6_test.npy")
    y_train=np.load("processed/SMD/machine-1-6_labels.npy")
    
    print(X_train.shape,X_test.shape)   
    print(y_train.shape)
    print("##########")
    y_train_modified=[]
    for i in range(len(y_train)):
        # print(y_train[i])
        if 1 in y_train[i]:
            y_train_modified.append(1)
        else:
            y_train_modified.append(0)

        # break
    # print(np.array(y_train_modified)[240:255])
    print("##########")
    
    # print(np.allclose(X_train,X_test))
    # print(X_train)
    # print(X_test)
    print(np.sum(y_train_modified))
    

    np.savetxt("Dataset_AD/smd_dataset_train.txt",X_train)
    np.savetxt("Dataset_AD/smd_dataset_test.txt",X_test)
    np.savetxt("Dataset_AD/smd_dataset_label.txt",y_train_modified)
    # print(y_train)
    
# create_smd_dataset()


#SWAT
def create_swat_dataset():

    X_train=np.load("processed/SWaT/train.npy")
    X_test=np.load("processed/SWaT/test.npy")
    y_train=np.load("processed/SWaT/labels.npy")
    
    print(X_train.shape,X_test.shape)   
    print(y_train.shape)
    print(X_train[:5],X_test[:5],y_train[:5])
    print("##########")
    print(y_train.sum())
    # plt.scatter(X_test,y_train)
    # plt.savefig("deleteme")

    np.savetxt("Dataset_AD/swat_dataset_train.txt",X_train)
    np.savetxt("Dataset_AD/swat_dataset_test.txt",X_test)
    np.savetxt("Dataset_AD/swat_dataset_label.txt",y_train)

# create_swat_dataset()

def create_ucr_dataset():


    X_train=np.load("processed/UCR/138_train.npy")
    X_test=np.load("processed/UCR/138_test.npy")
    y_train=np.load("processed/UCR/138_labels.npy")
    
    print(X_train.shape,X_test.shape)   
    print(y_train.shape)
    print(np.sum(y_train))
    print(X_train[:5],X_test[:5],y_train[:5])
    print("##########")

    plt.plot(X_train)
    plt.savefig("deleteme")

# create_ucr_dataset()


def create_msl_dataset():


    X_train=np.load("processed/SMAP/A-4_train.npy")
    X_test=np.load("processed/SMAP/A-4_test.npy")
    y_train=np.load("processed/SMAP/A-4_labels.npy")
    print("################")
    print(X_train.shape,X_test.shape)   
    print(y_train.shape)
    print(np.sum(y_train))
    print(np.unique(y_train))
    print(X_train[:5],X_test[:5],y_train[:5])
    print("################")

    plt.plot(X_train)
    plt.savefig("deleteme")






# create_msl_dataset()


# X_train=pd.read_csv("data/synthetic/synthetic_data_with_anomaly-s-1.csv")
# print(X_train)
# X_test=pd.read_csv("data/synthetic/test_anomaly.csv")
# print(X_test.head(5))

X_train=np.load("processed/UCR/138_train.npy")
X_test=np.load("processed/UCR/138_test.npy")
label=np.load("processed/UCR/138_labels.npy")

print(X_train,X_test,label)
print(X_train.shape,X_test.shape,label.shape)
print(label.sum())

# plt.plot(X_train)
# plt.savefig("xtrain")
# plt.show()
plt.plot(X_test)
plt.scatter([1187, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196],[0.57894784,0.67425138, 0.7101119,  0.73616817, 0.72021189, 0.68334315,
 0.62411302, 0.5548194,  0.48108501, 0.42414653],color='red')
plt.savefig("xtest")

print(X_test[np.where(label==1)])
print(np.where(label==1))


# y_train=np.load("processed/MBA/labels.npy")
# print("################")
# print(X_train.shape,X_test.shape)   
# print(y_train.shape)
# print(np.sum(y_train))
# print(np.unique(y_train))
# print(X_train[:5],X_test[:5],y_train[:5])
# print(y_train.sum())
# print("################")





