import os
import torch as pt
import data_setup, engine, model_builder,utils
from pathlib import Path

from torchvision import transforms

from sys import path
level=1
path_chosen = path[0]
print(path_chosen)
base_path = Path("/".join(map(lambda x:str(x),(path_chosen.split("/")[:-level]))))
print("\n",base_path)
modular_path = base_path / 'going_modular'
dataset_path = base_path / 'data/pizza_steak_sushi'

# Setup hyperparameters
NUM_EPOCHS = 20
BATCH_SIZE = 32
HIDDEN_UNITS = 20
LEARNING_RATE = 0.001
NUM_WORKERS = 0 #round(os.cpu_count()*(3/4))


train_dir = dataset_path/'train'
test_dir = dataset_path/'test'
print("\n",train_dir, "\n")
# Setup device
device = 'cuda' if pt.cuda.is_available() else 'cpu'

data_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, classes = data_setup.create_dataloaders(train_dir=train_dir, test_dir=test_dir, transform=data_transform, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

model = model_builder.TinyVGG(input_shape = 3, hidden_units=HIDDEN_UNITS, output_shape= len(classes)).to(device)

loss_fn = pt.nn.CrossEntropyLoss()

optimizer = pt.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)



engine.train(
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    epochs=NUM_EPOCHS,
    device=device
)

# Save model using  utils.py
utils.save_model(model, str(base_path/'models'),model_name='05_going_modular_script_mode_tinyvgg_model.pth')