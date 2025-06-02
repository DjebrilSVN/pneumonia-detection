import torch
import torchvision.models as models
import torch.nn as nn

def save_modeldep(file_name, model):
    save_dict = {
        'model_state_dict': model.state_dict(),
        'class_names': ['NORMAL', 'PNEUMONIA'],
        'input_size': 224
    }
    torch.save(save_dict, file_name)
    print(f"Model saved to {file_name}")


def load_model(model_path):
    try:
        # Recreate model architecture
        model = models.densenet161(pretrained=False)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 1)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")
