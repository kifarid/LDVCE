from torchvision import transforms
from data.datasets import CUB
import torch
from utils.preprocessor import ResizeAndNormalizer
from utils.cub_inception import CUBInception
from tqdm import tqdm
from torch.utils.data import DataLoader

n_epochs = 1000
train_path = '/misc/lmbraid21/schrodi/concepts/data/cub/CUB_processed/class_attr_data_10/train.pkl'
val_path = '/misc/lmbraid21/schrodi/concepts/data/cub/CUB_processed/class_attr_data_10/val.pkl'
test_path = '/misc/lmbraid21/schrodi/concepts/data/cub/CUB_processed/class_attr_data_10/test.pkl'


device = torch.device("cuda")
train_transform = transform = transforms.Compose([
            transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
            transforms.RandomResizedCrop(299),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  # implicitly divides by 255
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
train_set = CUB([train_path, val_path], use_attr=False, batch_size=64, image_dir='/misc/lmbraid21/schrodi/concepts/data/cub/CUB_200_2011', no_img=False ,uncertain_label=False, n_class_attr=-1, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True, drop_last=True, num_workers=16)
resized_resol = int(299 * 256 / 224)
test_transform = transform = transforms.Compose([
    transforms.Resize((resized_resol, resized_resol)),
    transforms.ToTensor(),  # implicitly divides by 255
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
test_set = CUB([test_path], use_attr=False, batch_size=64, image_dir='/misc/lmbraid21/schrodi/concepts/data/cub/CUB_200_2011', no_img=False,uncertain_label=False, n_class_attr=-1, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False, drop_last=False, num_workers=16)
model = CUBInception(name="cub_inception_model")
#model.fit(device, train_loader, test_loader, "/misc/lmbraid21/schrodi/pretrained_models", patience=50, n_epoch=n_epochs)

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = CUB( 
    pkl_file_paths=['/misc/lmbraid21/schrodi/concepts/data/cub/CUB_processed/class_attr_data_10/test.pkl'],
    use_attr=False,
    no_img=True,
    uncertain_label=False,
    image_dir='/misc/lmbraid21/schrodi/concepts/data/cub/CUB_200_2011',
    n_class_attr=-1,
    transform=transform,
    shard=0,
    num_shards=1,
)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16)

device = torch.device("cuda")

classifier_model = CUBInception(name="inception_model")
classifier_model.load_state_dict(torch.load("/misc/lmbraid21/schrodi/pretrained_models/cub_inception_model.pt", map_location='cpu'), strict=False)
resized_resol = int(299 * 256 / 224)
classifier_model = ResizeAndNormalizer(classifier_model, resolution=(resized_resol, resized_resol))
classifier_model = classifier_model.to(device).eval()

train_acc = 0
with torch.inference_mode():
    for batch in tqdm(data_loader):
        image, label, _ = batch
        logits = classifier_model(image.to(device))
        pred = torch.argmax(logits, dim=-1).cpu()
        train_acc += torch.sum(pred == label)
final_acc = train_acc/len(dataset)
print(final_acc.item())

