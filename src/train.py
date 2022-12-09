  # 가중치 초기화
def initialize_weights(model):
    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
      nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
      nn.init.normal_(model.weight.data, 1.0, 0.02)
      nn.init.constant(model.bias.data, 0.0)
      
      
  def train(model, dataloader, optimizer, device):
  running_loss=0
  model.train()
  iterator = tqdm.tqdm(dataloader)
  for blur, sharp in iterator:
    blur=blur.to(device)
    sharp=sharp.to(device)
    preds=model(blur)
    
    optimizer.zero_grad()
    loss=nn.MSELoss()(torch.squeeze(preds), sharp)
    loss.backward()
    optimizer.step()
    running_loss+=loss.item()
    iterator.set_description(f"epoch{epoch+1} train loss:{loss.item()}")
  return running_loss/len(dataloader)

def evaluate(model, dataloader, device):
  running_loss=0
  model.eval()
  iterator = tqdm.tqdm(dataloader)
  with torch.no_grad():
    for blur, sharp in iterator:
      blur=blur.to(device)
      sharp=sharp.to(device)
      preds=model(blur)
      loss=nn.MSELoss()(torch.squeeze(preds), sharp)
      running_loss+=loss.item()
      iterator.set_description(f"epoch{epoch+1} val loss:{loss.item()}")
  return running_loss/len(dataloader)

