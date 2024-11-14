import torch
import torch.nn as nn
import torchvision

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.utils.data import random_split, DataLoader, TensorDataset
import shap


num_classes = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
target_model = torchvision.models.resnet18(pretrained=False)  # You can use pretrained=True for a pre-trained model
target_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
target_model.fc = nn.Linear(target_model.fc.in_features, num_classes) 
target_model.load_state_dict(torch.load('resnet18_fmnist.pth'))  # Load your trained model weights
target_model = target_model.to(device)
target_model.eval() 


transform = transforms.Compose([
    transforms.Resize(224),  
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),  # Normalize grayscale images
])

batch_size = 100
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
# Define member and non-member DataLoaders
member_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)  # Member data (train data)
non_member_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)  # Non-member data (test data)


hidden_activations = []

# Define a function to save the activations from a specific layer
def save_activation(module, input, output):
    hidden_activations.append(output)

# Register the hook on the penultimate layer of the model
target_model.layer4[1].register_forward_hook(save_activation)


def get_hidden_activations(model, data_loader, device):
    model.eval()
    all_activations = []
    all_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            
            # Clear the activations list
            global hidden_activations
            hidden_activations = []
            
            # Forward pass
            _ = model(images)  # Only need to run forward to trigger the hook
            
            # Retrieve the activations from the hook
            #activations = hidden_activations[0]
            activations = hidden_activations[0].view(images.size(0), -1)
            
            all_activations.append(activations.cpu())
            all_labels.append(labels)
            
    return torch.cat(all_activations), torch.cat(all_labels)


class AttackModel(nn.Module):
    def __init__(self, input_dim):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # Binary output (member or non-member)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x

# Initialize attack model
hidden_activation_dim = 512*7*7 
attack_model = AttackModel(input_dim=hidden_activation_dim)
attack_model = attack_model.to(device)



member_activations, member_labels = get_hidden_activations(target_model, member_loader, device)
non_member_activations, non_member_labels = get_hidden_activations(target_model, non_member_loader, device)

# Create labels for attack training
member_labels = torch.ones(member_activations.size(0))  # Label 1 for members
non_member_labels = torch.zeros(non_member_activations.size(0))  # Label 0 for non-members

# Combine and create the final dataset
attack_data = torch.cat([member_activations, non_member_activations])
attack_labels = torch.cat([member_labels, non_member_labels])





# attack_dataset = TensorDataset(attack_data, attack_labels)
# attack_loader = DataLoader(attack_dataset, batch_size=64, shuffle=True)

# # Loss and optimizer
# criterion = nn.BCELoss()
# optimizer = optim.Adam(attack_model.parameters(), lr=1e-4)

# # Training loop
# num_epochs = 10
# attack_model.train()
# for epoch in range(num_epochs):
#     for batch_activations, batch_labels in attack_loader:
#         batch_activations, batch_labels = batch_activations.to(device), batch_labels.to(device)

#         # Forward pass
#         outputs = attack_model(batch_activations)
#         loss = criterion(outputs.squeeze(), batch_labels)

#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


attack_model.load_state_dict(torch.load("attack_model_weights.pth"))
attack_model.eval()  # Set the model to evaluation mode
print("Attack model weights loaded successfully.")




def save_activation(module, input, output):
    print(f"Hook called on module: {module}")
    hidden_activations.append(output)

# Register the hook again on the target model
target_model.layer4[1].register_forward_hook(save_activation)

class CombinedModel(nn.Module):
    def __init__(self, target_model, attack_model):
        super(CombinedModel, self).__init__()
        self.target_model = target_model
        self.attack_model = attack_model
    
    def forward(self, x):
        global hidden_activations
        hidden_activations = []  # Clear activations before each forward pass
        
        # Forward pass through the target model to generate hidden features
        _ = self.target_model(x)  # This triggers the hook to store hidden activations
        
        # Check if hook captured activations
        if not hidden_activations:
            raise RuntimeError("Hook did not capture any activations.")
        
        # Flatten the hidden features for input to the attack model
        hidden_features = hidden_activations[0].view(x.size(0), -1)
        
        # Pass hidden features through the attack model
        attack_output = self.attack_model(hidden_features)
        
        return attack_output

# Test the CombinedModel with a sample input
combined_model = CombinedModel(target_model, attack_model).to(device)

model = CombinedModel(target_model,attack_model)

X_test = next(iter(test_loader))[0]  # This will give a batch of test images
# X_test = X_test.permute(0, 2, 3, 1).cpu().numpy()  # Convert PyTorch tensor to numpy (HWC format)

masker = shap.maskers.Image("inpaint_telea", X_test[0].shape)


def f(x):
    # Convert input from HWC (batch_size, height, width, channels) to CHW (batch_size, channels, height, width)
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)  # HWC -> CHW
    print("Input shape to model:", x_tensor.shape)  # Expected: [batch_size, 1, 224, 224]
    with torch.no_grad():
        return model(x_tensor).cpu().numpy()



# Class names for Fashion MNIST
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Create the SHAP explainer with the model and image masker
explainer = shap.Explainer(f, masker, output_names=class_names)

# Explain two images using SHAP
shap_values = explainer(X_test[50:51], max_evals=100, batch_size=1)

# Plot the SHAP values for the images
shap.image_plot(shap_values, X_test[50:51])