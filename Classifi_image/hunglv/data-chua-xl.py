from PIL import Image
import numpy as np
import csv
import pprint, pickle


#doc file training
root_path ="/home/hunglv/PycharmProjects/Classification of traffic signs/data/Training/"

X_train = np.zeros((32,32,3), dtype=np.int64)
Y_train = np.zeros((1), dtype=np.int64)
print(X_train.shape)
X_train = np.array([X_train])

vk=0
for i in range(0,43):
    c = format(i, "05d")
    # print(c)
    path_folder = root_path+c+'/'
    path_csv = path_folder+'GT-'+c+'.csv'
    open_file = open(path_csv)
    GT_file = csv.reader(open_file,delimiter=';')
    # file_name = GT_file[0]
    # print(GT_file)
    for j in GT_file :
        print(str(i)+" : " + str(vk))
        vk = vk + 1
        file_name = j[0]
        if (j[0]!="Filename"):
           # print(file_name)
           path_file = path_folder+file_name
           im = Image.open(path_file)
           im = im.resize((32, 32), Image.ANTIALIAS)
           img = np.array(im)
           img = np.array([img])
           id_class =int( j[7])
           # print(img.shape)
           X_train = np.concatenate((X_train,img), axis = 0)
           Y_train = np.concatenate((Y_train,[id_class]), axis= 0)
X_train = np.delete(X_train,0,0)
Y_train = np.delete(Y_train,0)
print(X_train.shape)
print(Y_train.shape)
train = {'X': X_train, 'Y': Y_train}
print(train)
print("cdvjhs")
# doc file test
# path_file_anh = "/home/phung/PycharmProjects/traffic-dl-san/data-chua-xl/GTSRB _test/Final_Test/Images/"
# path_file_test ="/home/phung/PycharmProjects/traffic-dl-san/data-chua-xl/GTSRB _test/Final_Test/Images/GT-final_test.csv"
# open_file = open(path_file_test)
# GT_test = csv.reader(open_file, delimiter=';')
# X_test = np.zeros((30,30,3), dtype=np.int64)
# Y_test = np.zeros((1,1), dtype=np.int64)
# X_test = np.array([X_test])
# i=0
# for j in GT_test:
#     i= i+1
#     print(i)
#     if (j[0] !="Filename"):
#         name_file = j[0]
#         path_file = path_file_anh+name_file
#         im = Image.open(path_file)
#         im = im.resize((30,30),Image.ANTIALIAS)
#         img = np.array(im)
#         img = np.array([img])
#         id_class = int(j[7])
#         X_test = np.concatenate((X_test,img), axis = 0)
#         Y_test = np.concatenate((Y_test,[[id_class]]), axis = 0)
# X_test = np.delete(X_train, 0,0)
# Y_test = np.delete(Y_test,0,0)
# # print(X_test)
# # print(Y_test)
# print(X_test.shape)
# print(Y_test.shape)
# test = {'X':X_test, 'Y':Y_test}

######################## ghi file train

output_train = open('train.pkl','wb')
pickle.dump(train,output_train,-1)
output_train.close()
pkl_file = open('train.pkl', 'rb')

data1 = pickle.load(pkl_file)
pprint.pprint(data1)
Y_tran = np.array(data1['Y'])
print(Y_tran.shape)
print(Y_train.shape)
pkl_file.close()




##### ghi file test
# output_test = open('test.pkl','wb')
# pickle.dump(test,output_test)
#
# output_test.close()
# # pickle.dump()
# pkl_file2 = open('test.pkl','rb')
# data2 = pickle.load(pkl_file2)
# pprint.pprint(data2)
# pkl_file2.close()
# # data2 = pickle.load(pkl_file)
# # pprint.pprint(data2)
# pkl_file.close()