import torch
import torch.nn as nn
import torch.optim as optim
from models.base_models import ViTTiny
from data.data_loading import get_ants_bees_loaders

def main():
    # Get a single batch
    train_loader, _ = get_ants_bees_loaders(batch_size=8, img_size=224, num_workers=0)
    images, labels = next(iter(train_loader))
    
    print(f"Batch labels: {labels.tolist()}")
    print(f"Batch shape: {images.shape}")
    
    # Create model
    model = ViTTiny(num_classes=2, img_size=224, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Try to overfit this batch
    print("Attempting to overfit a single batch:")
    for epoch in range(100):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0) * 100
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/100, Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
    
    # Final prediction
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0) * 100
        print(f"Final accuracy on batch: {accuracy:.2f}%")
        print(f"Predictions: {predicted.tolist()}")
        print(f"Ground truth: {labels.tolist()}")

if __name__ == "__main__":
    main()