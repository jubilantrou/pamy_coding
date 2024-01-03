import pickle
import torch
import torch.nn as nn
from torch.nn.functional import dropout
import numpy as np
import math
import matplotlib.pyplot as plt
from CNN import CNN
# %% constant
num_traj = 3
q_1 = 25
q_2 = 2
mid_position = q_2
filter_size = 3
channel = 4
width = 1
# length of inputs
length_1 = 2 * q_1 + 1
# length of labels
length_2 = 2 * q_2 + 1
k = 0.7
batch_size = 50
learning_rate = 1e-4
weight_decay = 0.00
dropout = 0.01
epoches = 200
# %%
# generate data
vel = np.array([])
vel = np.append(vel, 0)
acc = np.array([])
acc = np.append(acc, 0)

suffix = ".txt"
path_of_file = "/home/hma/PyExe/Python/CNN/" + "TrainingData" + suffix
file = open(path_of_file, 'wb')

for i in range(num_traj):
    path_of_file_2 = "/home/hma/PyExe/Python/CNN/" + str(i) + suffix
    file_2 = open(path_of_file_2, 'rb')
    t_stamp_u = pickle.load(file_2) # time stamp for x-axis
    y_history = pickle.load(file_2) 
    ff_history = pickle.load(file_2) 
    file_2.close()

    for j in range( len(y_history[0]) ):
        if j > 0:
            vel = np.append(vel, (y_history[0][j] - y_history[0][j-1]) / 0.01 )

    for j in range( len(y_history[0]) ):
        if j > 0:
            acc = np.append(acc, (vel[j] - vel[j-1]) / 0.01 )

    pickle.dump( (y_history[0] - y_history[0][0]), file, -1) # time stamp for x-axis
    pickle.dump(vel, file, -1 )
    pickle.dump(acc, file, -1 )
    pickle.dump(ff_history[0] / 1000, file, -1)
    pickle.dump((ff_history[-1])/ 1000, file, -1)
    
file.close()
# %%
# read all the data
theta = np.random.rand( length_1 ) / 10
bias = np.random.rand( 1 ) / 10

dataset = []
labelset = []

train_data = []
train_label = []
val_data = []
val_label = []
y_eval = []
u_eval = np.array([])
y_eval_0 = []
u_eval_0 = np.array([])
y_eval_1 = []
u_eval_1 = np.array([])


file = open(path_of_file, 'rb')

for n in range(num_traj):
    # disired trajectory
    y = pickle.load(file)
    vel = pickle.load(file)
    acc = pickle.load(file)
    # the initial feedforward control
    u_ini = pickle.load(file)
    # the final feedforward control
    u = pickle.load(file)

    l = len(y)
    if n == 0:
        l_of_y_0 = l

    if n == 1:
        l_of_y_1 = l

    if n == 2:
        l_of_y = l

    for i in range(l):
        y_temp = []
        u_temp = []
        u_ini_temp = []
        vel_temp = []
        acc_temp = []
        for j in range(i-q_1, i+q_1+1):
            if j < 0:
                y_temp.append( y[0] )
                u_ini_temp.append( 0 )
                vel_temp.append( 0 )
                acc_temp.append( 0 )
            elif j > l-1:
                y_temp.append( y[l-1] )
                u_ini_temp.append( u_ini[l-1] )
                vel_temp.append( vel[l-1] )
                acc_temp.append( acc[l-1] )
            else:
                y_temp.append( y[j] )
                u_ini_temp.append( u_ini[j] )
                vel_temp.append( vel[j] )
                acc_temp.append( acc[j] )
        
        for j in range(i-q_2, i+q_2+1):
            if j < 0:
                u_temp.append( 0 )
            elif j > l-1:
                u_temp.append( u[l-1] )
            else:
                u_temp.append( u[j] )
        
        # labelset.append( np.asscalar(bias + np.dot( np.array(y_temp), theta )) )
        if n == 0:
            # y_eval_0.append( torch.tensor(y_temp, dtype=float).view(1, channel, length_1) )
            y_eval_0.append( torch.tensor(y_temp + vel_temp + acc_temp + u_ini_temp, dtype=float).view(1, channel, length_1) )
            # u_eval = np.append(u_eval, np.asscalar(bias + np.dot( np.array(y_temp), theta )) )
            u_eval_0 = np.append(u_eval_0, np.asscalar( u[i] ))

        if n == 1:
            # y_eval_1.append( torch.tensor(y_temp, dtype=float).view(1, channel, length_1) )
            y_eval_1.append( torch.tensor(y_temp + vel_temp + acc_temp + u_ini_temp, dtype=float).view(1, channel, length_1) )
            # u_eval = np.append(u_eval, np.asscalar(bias + np.dot( np.array(y_temp), theta )) )
            u_eval_1 = np.append(u_eval_1, np.asscalar( u[i] ))

        if n == 2:
            # batch_size * channel * length
            # y_eval.append( torch.tensor(y_temp, dtype=float).view(1, channel, length_1) )
            y_eval.append( torch.tensor(y_temp + vel_temp + acc_temp + u_ini_temp, dtype=float).view(1, channel, length_1) )
            # u_eval = np.append(u_eval, np.asscalar(bias + np.dot( np.array(y_temp), theta )) )
            u_eval = np.append(u_eval, np.asscalar( u[i] ))
        
        labelset.append( u_temp )
        # dataset.append( y_temp)
        dataset.append( [y_temp, vel_temp, acc_temp, u_ini_temp] )

file.close()

num_point = len( labelset )
num_train = int( k * num_point )
num_val = num_point - num_train

arr = np.arange( num_point )
# np.random.shuffle( arr )

for i in range( num_point ):
    if i < num_train:
        train_data.append( dataset[arr[i]] )
        train_label.append( labelset[arr[i]] )
    else:
        val_data.append( torch.tensor(dataset[arr[i]], dtype=float ).view(1, channel, length_1)) 
        val_label.append( torch.tensor(labelset[arr[i]], dtype=float ).view(1, length_2))


train_loader = []

idx = 0
while 1:
    if idx + batch_size - 1 < num_train:
        data_temp = train_data[idx : idx+batch_size]
        label_temp = train_label[idx : idx+batch_size]
        batch_x = torch.tensor(data_temp, dtype=float).view(batch_size, channel, length_1)
        batch_y = torch.tensor(label_temp, dtype=float).view(batch_size, length_2)
        train_loader.append( (batch_x, batch_y) )
        idx += batch_size
    else:
        break

# %%
# build the CNN model

iter_per_epoch = len(train_loader)
train_loss_history = []
val_loss_history = []

cnn_model = CNN( channels=channel, dropout=dropout, length_data=length_1, length_label=length_2, filter_size=filter_size )
print(cnn_model)

optimizer = torch.optim.Adam(cnn_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_function = nn.L1Loss(size_average=True)


for epoch in range(epoches):

    avg_train_loss = 0.0 # loss summed over epoch and averaged
    cnn_model.train()

    for (batch_data, batch_label) in train_loader:
        output = cnn_model(batch_data.float())
        loss = loss_function(output.squeeze(), batch_label.squeeze().float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_train_loss += loss.item()
    
    avg_train_loss /= iter_per_epoch
    train_loss_history.append(avg_train_loss) 
    # print ('\n[Epoch {}/{}] TRAIN loss: {:.3f}'.format(epoch+1, epoches, avg_train_loss))

    avg_eval_loss = 0.0
    cnn_model.eval()

    for i in range(len(val_label)):      
        preds = cnn_model(val_data[i].float())
        loss = loss_function(preds.squeeze(), val_label[i].float())
        avg_eval_loss += loss.item()
 
    avg_eval_loss /= len(val_label)
    val_loss_history.append(avg_eval_loss)
    print ('\n[Epoch {}/{}] TRAIN/VALID loss: {:.3}/{:.3f}'.format(epoch+1, epoches, avg_train_loss, avg_eval_loss))


# %%
plt.figure(1)
plt.plot(train_loss_history, '-o', label='Training')
plt.plot(val_loss_history, '-o', label='Evaluation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(ncol=2, loc='upper center')
plt.show()

# %%
u_model = np.array([])
t_stamp = np.linspace(0, (l_of_y-1)/100, num=l_of_y, endpoint=True)

for data_y in y_eval:
    output = cnn_model(data_y.float())
    u_model = np.append(u_model, np.asscalar(output[0][mid_position]) )

plt.figure(2)
plt.plot(t_stamp, u_model, linewidth=1, label='Prediction')
plt.plot(t_stamp, u_eval, linewidth=1, label='Desired')
plt.xlabel('t')
plt.ylabel('Control')
plt.legend(ncol=2, loc='upper center')
plt.show()
    
# %%
u_model_0 = np.array([])
t_stamp_0 = np.linspace(0, (l_of_y_0-1)/100, num=l_of_y_0, endpoint=True)

for data_y in y_eval_0:
    output = cnn_model(data_y.float())
    u_model_0 = np.append(u_model_0, np.asscalar(output[0][mid_position]) )

plt.figure(3)
plt.plot(t_stamp_0, u_model_0, linewidth=1, label='Prediction')
plt.plot(t_stamp_0, u_eval_0, linewidth=1, label='Desired')
plt.xlabel('t')
plt.ylabel('Control')
plt.legend(ncol=2, loc='upper center')
plt.show()

# %%
u_model_1 = np.array([])
t_stamp_1 = np.linspace(0, (l_of_y_1-1)/100, num=l_of_y_1, endpoint=True)

for data_y in y_eval_1:
    output = cnn_model(data_y.float())
    u_model_1 = np.append(u_model_1, np.asscalar(output[0][mid_position]) )

plt.figure(4)
plt.plot(t_stamp_1, u_model_1, linewidth=1, label='Prediction')
plt.plot(t_stamp_1, u_eval_1, linewidth=1, label='Desired')
plt.xlabel('t')
plt.ylabel('Control')
plt.legend(ncol=2, loc='upper center')
plt.show()  