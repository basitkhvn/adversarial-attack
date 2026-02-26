from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import io
from PIL import Image
from torchvision import transforms
import base64
from fgsm import Attack 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x # Returns raw logits (scores)
    
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN().to(device)
path='mnist_model.pth'
saved_states=torch.load(path,weights_only=True)
model.load_state_dict(saved_states)
model.eval()

def imageprocess(bytes):
    image = Image.open(io.BytesIO(bytes))
    image= image.convert('L')
    image= image.resize((28,28)) # 28 BECAUSE OF MNIST IMAGE SIZES
    transform=transforms.ToTensor()
    tensor=transform(image)
    tensor = tensor.unsqueeze(0).to(device)
    return tensor

@app.post("/attack")
async def attackpoint(file: UploadFile = File(...),epsilon: float = Form(0.1)):
    bytes= await file.read()
    data=imageprocess(bytes)
    data.requires_grad=True #track gradient for the tensor

    # this is for the initial step we will check the correct output, because we dont know the target
    output = model(data)
    init_pred=output.max(1,keepdim=True)[1].item()
    target = torch.tensor([init_pred]).to(device)
    loss = nn.CrossEntropyLoss()(output, target)
    #now manipulate the image
    model.zero_grad()
    loss.backward()
    gradient=data.grad.data
    #attack
    attacker= Attack(model,device)
    changed_img=attacker.attack_function(epsilon,data,gradient)
    finaloutput=model(changed_img)
    final_pred=finaloutput.max(1,keepdim=True)[1].item()
    #convert into base64
    img_tensor = changed_img.squeeze().detach().cpu() # Remove batch dim, move to CPU
    to_pil = transforms.ToPILImage()
    changed_img = to_pil(img_tensor)

    buff= io.BytesIO()
    changed_img.save(buff,format="PNG")
    img_str= base64.b64encode(buff.getvalue()).decode("utf-8")
    if(init_pred!=final_pred):
        status="Success"
    else:
        status="Failed"
    return {
        "original_prediction": init_pred,
        "adversarial_prediction": final_pred,
        "status": status,
        "image": img_str
    }