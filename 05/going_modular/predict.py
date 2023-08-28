import torch as pt
from sys import path
import torchvision
import matplotlib.pyplot as plt
from model_builder import TinyVGG
file_name = 'pizza1.jpg'

level=1
path_chosen = path[0]
base_path = "/".join(map(lambda x:str(x),(path_chosen.split("/")[:-level])))
file_path = f"{base_path}/predict_file/{file_name}"

def pred_and_plot_image(model:pt.nn.Module, image_path:str, device:pt.device, class_names: list[str] = None, transform: torchvision.transforms=None):
    """Makes a prediction on a target image with a trained model and plots the image and prediction"""
    target_image = torchvision.io.read_image(str(image_path)).type(pt.float32)

    # Devide the image pixel values by 255 to get them between 0-1
    target_image/=255.

    # Transform our image if necessary
    if transform:
        target_image = transform(target_image)

    model.to(device)
    model.eval()
    with pt.inference_mode():

        # Make a prediction on the image
        logits = model(target_image.to(device).unsqueeze(dim=0))
    # Convert the logits to pred probs
    pred_probs = pt.softmax(logits, dim=1)

    # Convert our prediction probabilities -> prediction labels
    target_image_pred_label = pt.argmax(pred_probs, dim=1)

    # Plot our image alongside the prediction and prediction probability
    plt.imshow(target_image.permute(1,2,0))
    if class_names:
        title=f'Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {pred_probs.max().cpu():.3f}'
    else:
        title=f'Pred: {target_image_pred_label.cpu().item()} | Prob: {pred_probs.max().cpu():.3f}'
    plt.title(title)
    plt.axis(False)
    plt.show()
    
model = TinyVGG(3,10,3)
prepare_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([64,64], antialias=None),
])
pred_and_plot_image(
    model=model,
    image_path=file_path,
    device='cpu',
    class_names=['pizza', 'steak', 'sushi'],
    transform=prepare_transform
)