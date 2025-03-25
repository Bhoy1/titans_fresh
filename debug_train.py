import torch
import torch.nn as nn
import torch.optim as optim
from models.base_models import ViTTiny
from data.data_loading import get_ants_bees_loaders
from tqdm import tqdm

def main():
    # Get loaders with smaller batch size
    train_loader, val_loader = get_ants_bees_loaders(batch_size=16, img_size=224, num_workers=0)
    
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    
    # Create model
    model = ViTTiny(num_classes=2, img_size=224, pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Debug epoch function with detailed logging
    def debug_epoch(model, loader, optimizer=None, is_training=True):
        if is_training:
            model.train()
        else:
            model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.set_grad_enabled(is_training):
            for batch_idx, (inputs, targets) in enumerate(tqdm(loader)):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                if is_training:
                    # Backward and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(targets).sum().item()
                correct += batch_correct
                total += targets.size(0)
                
                all_preds.extend(predicted.cpu().tolist())
                all_targets.extend(targets.cpu().tolist())
                
                if batch_idx % 2 == 0:
                    print(f"{'Train' if is_training else 'Val'} Batch {batch_idx}: "
                        f"Loss: {loss.item():.4f}, "
                        f"Batch Acc: {100 * batch_correct / targets.size(0):.2f}%, "
                        f"Running Acc: {100 * correct / total:.2f}%")
        
        # Analyze per-class performance
        if not is_training:
            class_correct = [0, 0]
            class_total = [0, 0]
            for pred, target in zip(all_preds, all_targets):
                class_correct[target] += int(pred == target)
                class_total[target] += 1
            
            for i in range(2):
                class_acc = 100 * class_correct[i] / max(1, class_total[i])
                print(f"Class {i} ({loader.dataset.classes[i]}): {class_acc:.2f}% "
                      f"({class_correct[i]}/{class_total[i]})")
        
        return total_loss / len(loader), 100 * correct / total
    
    # Train for just 5 epochs with extensive logging
    for epoch in range(5):
        print(f"\nEpoch {epoch+1}/5")
        train_loss, train_acc = debug_epoch(model, train_loader, optimizer, is_training=True)
        val_loss, val_acc = debug_epoch(model, val_loader, is_training=False)
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    print("Debug training complete")

if __name__ == "__main__":
    main()