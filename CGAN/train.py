import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import model

device = torch.device('mps')model_D = Discriminator().to(device)
model_G = Generator().to(device)

true_labels = torch.ones(BATCH_SIZE).reshape(BATCH_SIZE, 1).to(device)   #1のラベル
fake_labels = torch.zeros(BATCH_SIZE).reshape(BATCH_SIZE, 1).to(device)   #0のラベル

criterion = nn.BCELoss()    #損失関数はBCE(Binary Cross Entropy)を使用

optimizer_D = optim.Adam(model_D.parameters(), lr=0.00001)   #最適化関数はAdamを使用
optimizer_G = optim.Adam(model_G.parameters(), lr=0.00001)

epoch_num = 100   #エポック数（同じデータセットを何回学習するか）
print_coef = 10
G_train_ratio = 2   #識別器の学習1回あたり生成器が学習する回数
train_length = len(train_data)

def calc_acc(pred):   #正解率の評価関数
  pred = torch.where(pred > 0.5, 1., 0.)
  acc = pred.sum()/pred.size()[0]
  return acc

history = {"loss_D": [], "loss_G": [], "acc_true": [], "acc_fake": []}
n = 0
m = 0

for epoch in range(epoch_num):
  train_loss_D = 0
  train_loss_G = 0
  train_acc_true = 0
  train_acc_fake = 0

  model_D.train()
  model_G.train()
  for i, data in enumerate(train_loader):
    optimizer_D.zero_grad()
    inputs, labels = data[0].to(device), data[1].to(device)

    #識別器の学習（1）
    outputs = model_D(inputs, labels)   #本物の画像が本物か偽物か判定
    loss_true = criterion(outputs, true_labels)   #本物のデータを本物と判定するように学習したいので1のラベルを使用
    acc_true = calc_acc(outputs)

    #識別器の学習（2）
    noise = torch.randn((BATCH_SIZE, 100), dtype=torch.float32).to(device)   #ランダムな配列を生成
    noise_label = torch.from_numpy(np.random.randint(0,10,BATCH_SIZE)).clone().to(device)   #1~9のランダムな整数を生成
    inputs_fake = model_G(noise, noise_label).to(device)   #偽物の画像を生成
    outputs_fake = model_D(inputs_fake.detach(), noise_label)   #偽物の画像が本物か偽物か判定
    loss_fake = criterion(outputs_fake, fake_labels)   #偽物の画像を偽物と判定する方向に学習したいので0のラベルを使用
    acc_fake = calc_acc(outputs_fake)
    loss_D = loss_true + loss_fake   #識別器の学習（1）・（2）の損失を足し合わせる
    loss_D.backward()
    optimizer_D.step()   #識別器のパラメータを更新する

    #生成器の学習
    for _ in range(G_train_ratio):   #G_train_ratioの回数分生成器の学習を繰り返す
      optimizer_G.zero_grad()
      noise = torch.randn((BATCH_SIZE, 100), dtype=torch.float32).to(device)  #ランダムな配列を生成
      noise_label = torch.from_numpy(np.random.randint(0,10,BATCH_SIZE)).clone().to(device)   #1~9のランダムな整数を生成
      inputs_fake = model_G(noise, noise_label).to(device)
      outputs_fake = model_D(inputs_fake, noise_label)
      loss_G = criterion(outputs_fake, true_labels)   #本物と判定される偽物の画像を生成したいので1のラベルを使用
      loss_G.backward()
      optimizer_G.step()   #生成器のパラメータを更新する

    #学習経過の保存
    train_loss_D += loss_D.item()
    train_loss_G += loss_G.item()
    train_acc_true += acc_true.item()
    train_acc_fake += acc_fake.item()
    n += 1
    history["loss_D"].append(loss_D.item())
    history["loss_G"].append(loss_G.item())
    history["acc_true"].append(acc_true.item())
    history["acc_fake"].append(acc_fake.item())

    if i % ((train_length//BATCH_SIZE)//print_coef) == (train_length//BATCH_SIZE)//print_coef - 1:
      print(f"epoch:{epoch+1}  index:{i+1}  loss_D:{train_loss_D/n:.10f}  loss_G:{train_loss_G/n:.10f}  acc_true:{train_acc_true/n:.10f}  acc_fake:{train_acc_fake/n:.10f}")

      n = 0
      train_loss_D = 0
      train_loss_G = 0
      train_acc_true = 0
      train_acc_fake = 0

print("finish training")
