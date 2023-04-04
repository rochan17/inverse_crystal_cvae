import torch 
import torch.nn as nn
import numpy as np 
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchsummary import summary
# from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import seaborn as sns

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(dev)


### Defining VAE

class VAE(nn.Module):
    
    def __init__(self, num_channels):
        super(VAE, self).__init__()
        self.encoder=nn.Sequential(
            nn.Conv1d(in_channels=num_channels, out_channels=16, kernel_size=5, stride=1,padding=2),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            # nn.Conv1d(32, 64, kernel_size=3, stride=1),
            # nn.BatchNorm1d(64),
            # nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(2208,1024),
            nn.BatchNorm1d(1024),
            nn.Sigmoid()
        )

        self.mu = nn.Linear(1024, 100)
        self.logvar= nn.Linear(1024, 100)
        self.fc3 = nn.Linear(101,1024)
        self.fc4 = nn.Linear(1024,2208)

        self.decoder = nn.Sequential(
            nn.Unflatten(-1, (32,69)),
            nn.ConvTranspose1d(32, 16, kernel_size=3, stride=2,padding = 1),
            nn.LeakyReLU(0.2),
            # nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2),
            # nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16, num_channels, kernel_size=4,stride=1, padding = 1)
#             nn.Sigmoid()

        )
       
       
    
       
    def reparameterize(self,x, mu, logvar):
          std = torch.exp(0.5*logvar)
          # std = logvar.mul(0.5).exp_()
          esp = torch.randn(mu.size()).to(dev)
          z = mu + std*esp
          z_concat = torch.zeros(len(x),101).to(dev)
          for i in range(len(x)):
            z_concat[i] = torch.cat([z[i].reshape(1,100), x[i][-1,0].reshape(1,1)],1)
          return z_concat
    
    def bottleneck(self,x, h):
          mu, logvar = self.mu(h), self.logvar(h)
          # mu = torch.clamp(mu, min=-5, max=5)
          # logvar = torch.clamp(logvar, min=-1, max=1)
          z = self.reparameterize(x, mu, logvar)
          return z, mu, logvar

    def encode(self, x):
            h = self.encoder(x)
            z, mu, logvar = self.bottleneck(x,h)
            return z, mu, logvar

    def decode(self, z):
            y_pred = self.fc3(z)
            y_pred = self.fc4(y_pred)
            y_pred = self.decoder(y_pred)
            return y_pred

    def forward(self, x):
            z, mu, logvar = self.encode(x)
            y_pred = self.decode(z)
#             lattice = y_pred[:,0,:]
#             group = y_pred[:,1:,10:29]
#             period = y_pred[:,1:,0:10]
#             x=y_pred[:,1:,29:41]
#             y=y_pred[:,1:,41:53]
#             z_t=y_pred[:,1:,53:65]
#             prop1 = y_pred[:,1:,65:76]
#             prop2 = y_pred[:,1:,76:87]
#             prop3 = y_pred[:,1:,87:100]
#             prop4 = y_pred[:,1:,100:111]
#             prop5 = y_pred[:,1:,111:122]
#             prop6 = y_pred[:,1:,122:127]
#             prop7 = y_pred[:,1:,127:138]
#             x=F.softmax(x,dim=-1)
#             y=F.softmax(y,dim=-1)
#             z_t=F.softmax(z_t,dim=-1)
#             prop1=F.softmax(prop1,dim=-1)
#             prop2=F.softmax(prop2,dim=-1)
#             prop3=F.softmax(prop3,dim=-1)
#             prop4=F.softmax(prop4,dim=-1)
#             prop5=F.softmax(prop5,dim=-1)
#             prop6=F.softmax(prop6,dim=-1)
#             prop7=F.softmax(prop7,dim=-1)
#             group=F.softmax(group,dim=-1)
#             period=F.softmax(period,dim=-1)
#             lattice = torch.sigmoid(lattice)
#             lattice = lattice.reshape(-1,1,138)
#             #specie=self.species_mod(specie).view(img.shape[0],1,20,2)
#             xyz=torch.cat([period,group,x,y,z_t,prop1,prop2,prop3,prop4,prop5,prop6,prop7],dim=-1)
#             # diff=y_pred.shape[2]-6
#             y_pred=torch.cat([lattice,xyz],dim=1)
            return z, mu, logvar, y_pred

#     def forward(self,x):
#       return self.encoder(x)

trainloader = torch.utils.data.DataLoader(train, batch_size=128, shuffle=True, drop_last = True)


## Loss Function

def loss_fn(recon_x, x, mu, logvar, beta):
    # MSE = torch.mean((x-recon_x).pow(2))
    MSE = F.mse_loss(recon_x, x, reduction='sum')
#     MSE = F.mse_loss(recon_x[:,0,:], x[:,0,:], reduction='sum')
    KLD = -0.5*torch.sum(1 + logvar - mu.pow(2) - torch.exp(logvar)).cuda()
    # cos_proximity = loss_function(recon_x.flatten(), x.flatten())
#     Dice = DiceLoss(x[:,1:,:],recon_x[:,1:,:])
    return  MSE*(128/(128+2088))*500 +KLD*(2088/(128+2088))*1.2,MSE , KLD
#     return  MSE+ KLD*beta +Dice*50000, Dice, KLD*beta



## Training Step
epochs=40
beta =1 
for epoch in range(epochs):
  train_loss, train_recon , train_KLD = 0.0,0.0,0.0
#   validation_loss, validation_recon , validation_KLD = 0.0,0.0,0.0
  vae.train()
  for batch_idx,x in enumerate(trainloader):
#     with torch.autograd.detect_anomaly():
        x = x.to(dev)
        z, mu, logvar,recon_x = vae(x)
        # print(recon_x)
        # print(x)
        # print(x.shape)
        # print(z.shape)
        #print(recon_x)
        # print(logvar)
        # print(mu)
        # print(torch.isnan(recon_x).any())
        # print(torch.isnan(mu).any())
        # print(torch.isnan(logvar).any())
        loss, MSE, KLD = loss_fn(recon_x, x, mu, logvar, beta)
        # loss = loss_function(recon_x, x,target=labels).sum()
        # train_loss += loss.item()
        # loss = loss_function(x,recon_x)
        # if torch.isnan(loss):
        #  print('NaN Loss')
        # else:
        #  print('Loss: ', loss.item())
         
        # if torch.isnan(MSE):
        #  print('NaN MSE')
        # else:
        #   print('MSE: ', MSE.item())
        # if torch.isnan(KLD):
        #  print('NaN KLD')
        # else:
        #  print('KLD: ', KLD.item())
        train_loss += loss.item()
        train_recon += MSE.item()
        train_KLD += KLD.item()
        # train_Dice += Dice.item()
        optimizer.zero_grad()
        loss.backward()
        # # nn.utils.clip_grad_norm_(vae.parameters(), max_norm=2.0, norm_type=2)
        # nn.utils.clip_grad_value_(vae.parameters(), clip_value=1.0)
        optimizer.step()
        # for p in vae.parameters():
        #             with torch.no_grad():
        #                 print("Norm of gradient:", torch.sum(p.grad.data**2))
        #                 p.grad.zero_()
        # print(list(vae.parameters()))
        # print('------------------------------')
#   vae.eval()
#   for batch_idx,x in enumerate(validationloader):
#     x = x.to(dev)
#     z, mu, logvar,recon_x = vae(x)
#     loss, MSE, KLD = loss_fn(recon_x, x, mu, logvar, beta)
#     loss, MSE, KLD = loss_fn(recon_x, x, mu, logvar, beta)
#     validation_loss += loss.item()
#     validation_recon += MSE.item()
#     validation_KLD += KLD.item()
#     validation_Dice += Dice.item()

    # validate_performance()   
#   scheduler1.step()   
  scheduler2.step(train_loss) 
  print("\tEpoch", epoch + 1, "complete!", "\tTraining Loss: ", train_loss/len_dataset,train_recon/len_dataset,train_KLD/len_dataset)
#   print("\tEpoch", epoch + 1, "complete!", "\tValidation Loss: ", validation_loss/len_valid, validation_recon/len_valid, validation_KLD/len_valid)

  # print(list(vae.parameters()))
  print('------------------------------')
    
print("Finish!!")


      