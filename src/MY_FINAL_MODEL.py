class MY_FINAL_MODEL(nn.Module):
  def __init__(self):
    super(MY_FINAL_MODEL, self).__init__()
    self.conv1=nn.Conv2d(3, 16, kernel_size=3, padding=1)
    self.conv2=nn.Conv2d(16, 32, kernel_size=3, padding=1)
    self.conv3=nn.Conv2d(32, 64, kernel_size=3, padding=1)
    self.conv4=nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.conv5=nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.conv6=nn.Conv2d(16, 3, kernel_size=3, padding=1)
    
    self.relu=nn.LeakyReLU(0.01)

    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    self.upsample1=nn.ConvTranspose2d(256, 128, kernel_size=4, padding=1, stride=2) # k+(w−1)s−2p
    self.upsample2=nn.ConvTranspose2d(128, 64, kernel_size=4, padding=1, stride=2) # k+(w−1)s−2p    
    self.upsample3=nn.ConvTranspose2d(64, 32, kernel_size=4, padding=1, stride=2) # k+(w−1)s−2p    
    self.upsample4=nn.ConvTranspose2d(32, 16, kernel_size=4, padding=1, stride=2) # k+(w−1)s−2p
    self.upsample5=nn.ConvTranspose2d(16, 16, kernel_size=4, padding=1, stride=2) # k+(w−1)s−2p                
  

    self.bn1=nn.BatchNorm2d(16)
    self.bn2=nn.BatchNorm2d(32)    
    self.bn3=nn.BatchNorm2d(64)
    self.bn4=nn.BatchNorm2d(128)
    self.bn5=nn.BatchNorm2d(256)    

    self.bn6=nn.BatchNorm2d(128)
    self.bn7=nn.BatchNorm2d(64)    
    self.bn8=nn.BatchNorm2d(32)
    self.bn9=nn.BatchNorm2d(16)

    
  def forward(self,x):
    x=self.conv1(x) # 224*224*16
    x=self.bn1(x)
    x=self.relu(x)
    x=self.pool(x) # 112*112*16

    x0 = x
    x=self.conv2(x) # 112*112*32
    x=self.bn2(x)
    x=self.relu(x)
    x=self.pool(x) # 56*56*32
    x1 = x

    x=self.conv3(x) # 56*56*64
    x=self.bn3(x)
    x=self.relu(x)
    x=self.pool(x) # 28*28*64
    x2 = x

    x=self.conv4(x) # 28*28*128
    x=self.bn4(x)
    x=self.relu(x)
    x=self.pool(x) # 14*14*128

    x=self.upsample2(x) # 28*28*64
    x=self.bn7(x)
    x+=x2 # skip connection
    x=self.relu(x)

    x=self.upsample3(x) # 56*56*32
    x=self.bn8(x)
    x+=x1 # skip connection
    x=self.relu(x)

    x=self.upsample4(x) # 112*112*16
    x=self.bn9(x)
    x+=x0 # skip_connection
    x=self.relu(x)
    x=self.upsample5(x) # 224*224*16

    x=self.conv6(x) # 224*224*3
    return x
