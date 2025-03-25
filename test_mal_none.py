import torch
import torch.nn as nn
import torch.optim as optim
from models.model_factory import create_vit_with_memory
from data.data_loading import get_ants_bees_loaders
from configs.default_configs import DATASET_CONFIGS, MEMORY_CONFIGS, INTEGRATION_CONFIGS

def main():
    # Get dataset configs
    dataset_name = 'ants-bees'
    dataset_config = DATASET_CONFIGS[dataset_name]
    img_size = dataset_config['image_size']
    
    # Get memory and integration configs
    memory_type = 'none'
    approach = 'mal'
    memory_config = MEMORY_CONFIGS[memory_type]
    integration_config = INTEGRATION_CONFIGS[approach]
    
    # Load data
    train_loader, val_loader = get_ants_bees_loaders(batch_size=16, img_size=img_size, num_workers=0)
    
    # Create model using factory function
    model = create_vit_with_memory(
        model_size='tiny',
        approach=approach,
        memory_type=memory_type,
        num_classes=dataset_config['num_classes'],
        image_size=img_size,
        pretrained=True,
        **{**memory_config, **integration_config}
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")
    print(f"Created {approach.upper()} model with {memory_type} memory (size: tiny, pretrained: True)")
    
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Training loop
    for epoch in range(3):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            batch_correct = predicted.eq(targets).sum().item()
            train_correct += batch_correct
            train_total += targets.size(0)
            
            if batch_idx % 4 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss: {loss.item():.4f}, "
                      f"Batch Acc: {100 * batch_correct / targets.size(0):.2f}%, "
                      f"Running Acc: {100 * train_correct / train_total:.2f}%")
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(targets).sum().item()
                val_total += targets.size(0)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {100 * train_correct / train_total:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100 * val_correct / val_total:.2f}%\n")

if __name__ == "__main__":
    main()