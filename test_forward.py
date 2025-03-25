import torch
from models.base_models import ViTTiny

def main():
    # Create model
    model = ViTTiny(num_classes=2, img_size=224, pretrained=True)
    print("Created ViT-Tiny model")
    
    # Switch to eval mode
    model.eval()
    
    # Create a random input
    x = torch.randn(1, 3, 224, 224)
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    print(f"Output values: {output}")
    print(f"Predicted class: {output.argmax(dim=1).item()}")
    print(f"Class probabilities: {torch.softmax(output, dim=1)}")

if __name__ == "__main__":
    main()