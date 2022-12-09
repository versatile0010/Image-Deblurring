train_transform=transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transform=transforms.Compose([
          transforms.Resize((224, 224)),
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

gopro_local_path = '/content/drive/MyDrive/news/gopro/rename'

train_dataset = GOPRODATASET(path=gopro_local_path, train=True, transform=train_transform)
val_dataset =  GOPRODATASET(path=gopro_local_path, train=False, transform=val_transform)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, drop_last=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model=MY_FINAL_MODEL()
model.apply(initialize_weights)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

EPOCH = 100

train_loss=[]
val_loss=[]

best_loss=float('inf')
for epoch in range(EPOCH):
  train_loss_per_epoch=train(model, train_dataloader, optimizer, device)
  val_loss_per_epoch=evaluate(model, val_dataloader, device)

  if val_loss_per_epoch<best_loss:
    best_loss=val_loss_per_epoch
    torch.save(model.state_dict(), '/content/drive/MyDrive/news/deblur_model/1120/output/MY_FINAL_MODEL-best-7.pth')
  print(f"epoch{epoch+1} train loss : {train_loss_per_epoch} val loss : {val_loss_per_epoch} \n")
  print("===================================================================================\n")
  train_loss.append(train_loss_per_epoch)
  val_loss.append(val_loss_per_epoch)

torch.save(model.state_dict(), '/content/drive/MyDrive/news/deblur_model/1120/output/MY_FINAL_MODEL-done-7.pth')
print(f"\n============done ; best loss :  {best_loss} ===================\n")

x_list = range(1, EPOCH+1)

plt.title("train/validate loss")
plt.plot(x_list, train_loss, color='b', label="train_loss")
plt.plot(x_list, val_loss, color='r', label="val_loss")
plt.legend(shadow=True, fancybox=True, loc="upper right")
plt.xlabel('epoch') 
plt.ylabel('loss')
plt.show()
