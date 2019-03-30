import pandas as pd
import os

'''Root='F:/pc/senceClassification/UCMerced_LandUse'
list = os.listdir(Root)
namelist=[]
for i in range(0,len(list)):
    path = list[i]
    img_list=os.listdir(Root+'/'+ path)
    for imgname in img_list:
        namelist.append([Root + '/' + path +'/'+ imgname])
namelist=pd.DataFrame(namelist)
namelist.to_csv('imglist.csv',header=None,index=None)'''

table=pd.read_csv('imglist.csv',header=None,index_col=None)
table=table.sample(frac=1)
table[:210 * 8].to_csv('train.csv',header=None,index=None)
table[210 * 8:].to_csv('test.csv',header=None,index=None)