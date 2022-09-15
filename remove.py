import numpy as np
import os
# import 

# print(np.random.normal(0,1, size=(1)))
# return

# print("PROVA")
# loader=[]
# file="138_labels"
# loader.append(np.load(os.path.join("C:\\Users\\user\\Desktop\\TranAD\\processed\\UCR\\", f'{file}.npy')))

# file="138_test"
# loader.append(np.load(os.path.join("C:\\Users\\user\\Desktop\\TranAD\\processed\\UCR\\", f'{file}.npy')))

# file="138_train"
# loader.append(np.load(os.path.join("C:\\Users\\user\\Desktop\\TranAD\\processed\\UCR\\", f'{file}.npy')))

# print(len((loader[0])))
# print(len((loader[1])))
# print(len((loader[2])))
def create_synt_data():
    with open("test.txt",'w',encoding = 'utf-8') as f:
        f.write("timestamp,value\n")
        for year in [2013,2014]:
            for month in ["01","02","03","04","05","06","07","08","09","10","11","12"]:
                for day in ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22"]:
                    for hour in ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30"]:
                        for minutes in ["00","05","10","15","20","25","30","35","40","45","50","55"]:
                            f.write(str(year)+"-"+str(month)+"-"+str(day)+" "+str(hour)+":"+str(minutes)+":00"+","+str(np.random.normal(0,1,))+"\n")
                            # print(year,month,day,hour,minutes,"00",np.random.normal(0,1, size=(1)))

create_synt_data()

# print(loader)



