import torch
import torch.nn as nn
import torch.optim as optim
from models.model_factory import create_vit_with_memory
from data.data_loading import get_ants_bees_loaders
from configs.default_configs import DATASET_CONFIGS, MEMORY_CONFIGS, INTEGRATION_CONFIGS
from tqdm import tqdm

def main():
    # Parameters (matching the ones that worked)
    model_size = 'tiny'
    approach = 'mal'
    memory_type = 'none'
    batch_size = 16
    learning_rate = 0.0001
    epochs = 3  # Just enough to see if it works
    
    # Get dataset configs
    dataset_name = 'ants-bees'
    dataset_config = DATASET_CONFIGS[dataset_name]
    img_size = dataset_config['image_size']
    
    # Get memory and integration configs
    memory_config = MEMORY_CONFIGS[memory_type]
    integration_config = INTEGRATION_CONFIGS[approach]
    
    # Load data
    train_loader, val_loader = get_ants_bees_loaders(batch_size=batch_size, img_size=img_size, num_workers=0)
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    print(f"Class names: {train_loader.dataset.classes}")
    
    # Check class distribution
    train_targets = [sample[1] for sample in train_loader.dataset]
    class_counts = {}
    for i, cls in enumerate(train_loader.dataset.classes):
        class_counts[cls] = train_targets.count(i)
    print(f"Class distribution in training set: {class_counts}")
    
    # Create model using factory function - EXACTLY as in test_mal_none.py
    model = create_vit_with_memory(
        model_size=model_size,
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
    print(f"Created {approach.upper()} model with {memory_type} memory (size: {model_size}, pretrained: True)")
    
    # Setup loss and optimizer - EXACTLY as in test_mal_none.py
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Debug first batch
    debug_batch(model, train_loader, device, criterion, "initial")
    
    # Training loop
    for epoch in range(epochs):
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
        
        # Check class-wise accuracy
        class_correct = [0, 0]
        class_total = [0, 0]
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    class_correct[label] += (predicted[i] == targets[i]).item()
                    class_total[label] += 1
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {100 * train_correct / train_total:.2f}%")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {100 * val_correct / val_total:.2f}%")
        print("Class-wise accuracy:")
        for i in range(len(train_loader.dataset.classes)):
            print(f"  {train_loader.dataset.classes[i]}: {100 * class_correct[i] / class_total[i]:.2f}% ({class_correct[i]}/{class_total[i]})")
        
        # Debug a batch at the end of each epoch
        debug_batch(model, train_loader, device, criterion, f"epoch_{epoch+1}")

def debug_batch(model, loader, device, criterion, phase="debug"):
    """Debug the first batch of a data loader."""
    batch_data = next(iter(loader))
    inputs, targets = batch_data
    inputs, targets = inputs.to(device), targets.to(device)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        _, predicted = outputs.max(1)
    
    # Calculate metrics
    accuracy = 100 * predicted.eq(targets).sum().item() / targets.size(0)
    
    # Count class distribution in targets and predictions
    target_counts = {}
    pred_counts = {}
    for c in loader.dataset.classes:
        target_counts[c] = 0
        pred_counts[c] = 0
    
    for t in targets:
        target_counts[loader.dataset.classes[t.item()]] += 1
    
    for p in predicted:
        pred_counts[loader.dataset.classes[p.item()]] += 1
    
    # Print debug info
    print(f"\n===== {phase.upper()} DEBUG INFO =====")
    print(f"Batch size: {targets.size(0)}")
    print(f"Target distribution: {target_counts}")
    print(f"Prediction distribution: {pred_counts}")
    print(f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
    
    # Show individual predictions (up to 8 samples)
    print("Sample predictions (True -> Predicted, Correct?):")
    for i in range(min(8, targets.size(0))):
        t = loader.dataset.classes[targets[i].item()]
        p = loader.dataset.classes[predicted[i].item()]
        correct = "✓" if predicted[i] == targets[i] else "✗"
        print(f"  {t:8s} -> {p:8s} {correct}")
    
    print("=" * 50)
    
    # Return to train mode if needed
    model.train()
    return accuracy

if __name__ == "__main__":
    main()