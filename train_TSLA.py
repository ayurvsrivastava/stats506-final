from model import VolatilityModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import torch
import copy
import time
import matplotlib.pyplot as plt

df = pd.read_csv('TSLA_vols.csv')
strike = df['Contract Strike'].values
price = df['Close'].values
expiration = df['exp'].values
vol = df['vol'].values

X = np.array([strike, price, expiration]).T
y = np.array(vol)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = None
saved_model = False
# check if there is a saved model
if os.path.exists('tsla.pt'):
    # ask if we should load the model
    print("A model has been found. Do you want to load it? (y/n)") 
    answer = input()
    if answer == 'y':
        model = VolatilityModel()
        model.load_state_dict(torch.load('tsla.pt'))
        model.eval()
        saved_model = True
    else:
        model = VolatilityModel()
else:
    print("No saved model found")
    model = VolatilityModel()

criterion = torch.nn.MSELoss()
optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)

losses = []
lower_loss = 1000000
best_model = None
model.train()
if not saved_model:
    train_start = time.process_time_ns()
    for epoch in range(500):
        epoch_loss = []
        for i in range(len(X_train)):
            # get the inputs
            inputs = torch.from_numpy(X_train[i]).float()
            labels = torch.Tensor([y_train[i]])

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_loss.append(loss.item())
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, average loss: {np.mean(losses)}, lowest loss: {min(epoch_loss)}")
        if np.mean(losses) < lower_loss:
            print(f"Best model found at epoch {epoch} with mean loss {np.mean(losses)}")
            lower_loss = np.mean(losses)
            best_model = copy.deepcopy((epoch, model.state_dict(), optimizer.state_dict()))
    train_end = time.process_time_ns()
    print(f"Training took {(train_end - train_start) / 1e9} seconds")
    print('Finished Training')
    print(f"Best model found at epoch {best_model[0]} with loss {lower_loss}")
    model.load_state_dict(best_model[1])
    optimizer.load_state_dict(best_model[2])
    torch.save(model.state_dict(), 'tsla.pt')
    torch.save(optimizer.state_dict(), 'tsla_opt.pt')
else:
    print("Model loaded. Skipping training")
    model = VolatilityModel()
    model.load_state_dict(torch.load('tsla.pt'))

model.eval()

losses = []
time_per_sample = [None for _ in range(len(X_test))]
for i in range(len(X_test)):
    # get the inputs
    inputs = torch.from_numpy(X_test[i]).float()
    labels = torch.Tensor([y_test[i]])

    # forward + backward + optimize
    start_time = time.process_time_ns()
    outputs = model(inputs)
    end_time = time.process_time_ns()
    loss = criterion(outputs, labels)
    losses.append(loss.item())
    time_per_sample[i] = end_time - start_time

losses = np.array(losses)
stdev = np.std(losses)
mean = np.mean(losses)
losses = losses[losses < mean + 2 * stdev]
losses = losses[losses > mean - 2 * stdev]
print(f"Average loss: {np.mean(losses)}")

# remove outliers
time_per_sample = np.array(time_per_sample)
stdev = np.std(time_per_sample)
mean = np.mean(time_per_sample)
time_per_sample = time_per_sample[time_per_sample < mean + 2 * stdev]
time_per_sample = time_per_sample[time_per_sample > mean - 2 * stdev]
print(f"Average time per sample: {np.mean(time_per_sample) / 1e9}")

loss_per_expiration = {}
time_per_sample = [None for _ in range(len(X_test))]
for i in range(len(X_test)):
    # get the inputs
    inputs = torch.from_numpy(X_test[i]).float()
    labels = torch.Tensor([y_test[i]])

    # forward + backward + optimize
    start_time = time.process_time_ns()
    outputs = model(inputs)
    end_time = time.process_time_ns()
    loss = criterion(outputs, labels)
    loss_per_expiration[X_test[i][2]] = loss_per_expiration.get(X_test[i][2], []) + [loss.item()]
    time_per_sample[i] = end_time - start_time

for idx, key in enumerate(sorted(loss_per_expiration.keys())):
    loss_per_expiration[key] = np.array(loss_per_expiration[key])
    mean = np.mean(loss_per_expiration[key])
    stdev = np.std(loss_per_expiration[key])
    loss_per_expiration[key] = loss_per_expiration[key][loss_per_expiration[key] < mean + 2 * stdev]
    loss_per_expiration[key] = loss_per_expiration[key][loss_per_expiration[key] > mean - 2 * stdev]
    print(f"Average loss for expiration {key}: {np.mean(loss_per_expiration[key])}")
    plt.subplot(2, 2, idx + 1)
    plt.hist(loss_per_expiration[key], bins=75)
    plt.title(f"Expiration {round(key, 2)} months away")
    plt.xlabel("Loss")
    plt.ylabel("Frequency")
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.tight_layout()
plt.savefig('tsla_loss.png')
plt.show()

