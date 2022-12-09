class GOPRODATASET(Dataset):
  def __init__(self, path, train=True, transform=None):
    self.path=path
    if train:
      self.blur_path=path+"/train/blur/"
      self.sharp_path=path+"/train/sharp/"
    else: # test set
      self.blur_path=path+"/test/blur/"
      self.sharp_path=path+"/test/sharp/"
    self.blur_img_list=glob.glob(self.blur_path+'/*.png')
    self.sharp_img_list=glob.glob(self.sharp_path+'/*.png')
    self.blur_img_list.sort()
    self.sharp_img_list.sort()
    self.transform=transform
    self.total_img_list=self.blur_img_list+self.sharp_img_list
    self.input_img_list=self.blur_img_list

  def __len__(self):
    return len(self.input_img_list)

  def __getitem__(self, idx):
    blur_img_path=self.blur_img_list[idx]
    sharp_img_path=self.sharp_img_list[idx]

    blur_img=Image.open(blur_img_path)
    sharp_img=Image.open(sharp_img_path)

    if self.transform is not None:
      blur_img=self.transform(blur_img)
      sharp_img=self.transform(sharp_img)
    return blur_img, sharp_img
