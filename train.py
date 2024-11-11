#!/usr/bin/env python3
import os
import csv
import json
from tqdm import tqdm
import torch
import argparse
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model import MyModel
import numpy as np


class MiniPlaces(Dataset):
    def __init__(self, root_dir, split, transform=None, label_dict=None):
        """
        Initialize the MiniPlaces dataset with the root directory for the images,
        the split (train/val/test), an optional data transformation,
        and an optional label dictionary.

        Args:
            root_dir (str): Root directory for the MiniPlaces images.
            split (str): Split to use ('train', 'val', or 'test').
            transform (callable, optional): Optional data transformation to apply to the images.
            label_dict (dict, optional): Optional dictionary mapping integer labels to class names.
        """
        assert split in ['train', 'val', 'test']
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.filenames = []
        self.labels = []

        self.label_dict = label_dict if label_dict is not None else {}

        with open(os.path.join(self.root_dir, self.split + '.txt')) as r:
            lines = r.readlines()
            for line in lines:
                line = line.split()
                self.filenames.append(line[0])
                if split == 'test':
                    label = line[0]
                else:
                    label = int(line[1])
                self.labels.append(label)
                if split == 'train':
                    text_label = line[0].split('/')[2]
                    self.label_dict[label] = text_label

    def __len__(self):
        """
        Return the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Return a single image and its corresponding label when given an index.

        Args:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Tuple containing the image and its label.
        """
        if self.transform is not None:
            image = self.transform(
                Image.open(os.path.join(self.root_dir, "images", self.filenames[idx])))
        else:
            image = Image.open(os.path.join(self.root_dir, "images", self.filenames[idx]))
        label = self.labels[idx]
        return image, label


def create_train_transform():
    """
    Create training data transformation with augmentation
    """
    image_net_mean = torch.Tensor([0.485, 0.456, 0.406])
    image_net_std = torch.Tensor([0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=15,  # rotation
            translate=(0.1, 0.1),  # horizontal/vertical translation
            scale=(0.9, 1.1),  # scale
        ),
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize(image_net_mean, image_net_std)
    ])


def create_val_transform():
    """
    Create validation/test data transformation without augmentation
    """
    image_net_mean = torch.Tensor([0.485, 0.456, 0.406])
    image_net_std = torch.Tensor([0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize(image_net_mean, image_net_std)
    ])


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate the CNN classifier on the validation set.

    Args:
        model (CNN): CNN classifier to evaluate.
        test_loader (torch.utils.data.DataLoader): Data loader for the test set.
        criterion (callable): Loss function to use for evaluation.
        device (torch.device): Device to use for evaluation.

    Returns:
        float: Average loss on the test set.
        float: Accuracy on the test set.
    """
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        total_loss = 0.0
        num_correct = 0
        num_samples = 0

        for inputs, labels in test_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Compute the logits and loss
            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Compute the accuracy
            _, predictions = torch.max(logits, dim=1)
            num_correct += (predictions == labels).sum().item()
            num_samples += len(inputs)

    # Evaluate the model on the validation set
    avg_loss = total_loss / len(test_loader)
    accuracy = num_correct / num_samples

    return avg_loss, accuracy


def train(model, train_loader, val_loader, optimizer, criterion, device,
          num_epochs):
    """
    Train the CNN classifer on the training set and evaluate it on the validation set every epoch.

    Args:
    model (CNN): CNN classifier to train.
    train_loader (torch.utils.data.DataLoader): Data loader for the training set.
    val_loader (torch.utils.data.DataLoader): Data loader for the validation set.
    optimizer (torch.optim.Optimizer): Optimizer to use for training.
    criterion (callable): Loss function to use for training.
    device (torch.device): Device to use for training.
    num_epochs (int): Number of epochs to train the model.
    """

    # Place the model on device
    model = model.to(device)

    # Define early stopping parameters
    patience = 5  # Number of epochs to wait for improvement
    best_val_accuracy = 0.0  # Best validation accuracy so far
    epochs_without_improvement = 0  # Counter for epochs without improvement
    best_model_state = None  # To store the state of the best model

    # Performance tracking
    performance = []

    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        running_loss = 0.0  # Track cumulative loss for averaging
        correct_predictions = 0
        total_samples = 0

        with tqdm(total=len(train_loader),
                  desc=f'Epoch {epoch + 1}/{num_epochs}',
                  position=0,
                  leave=True) as pbar:
            for inputs, labels in train_loader:
                # Move inputs and labels to device
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the gradients
                optimizer.zero_grad()

                # Compute the logits and loss
                logits = model(inputs)
                loss = criterion(logits, labels)

                # Backward pass: Compute gradients
                loss.backward()

                # Optimize model parameters
                optimizer.step()

                # Track running loss
                running_loss += loss.item()

                # Track accuracy
                _, predicted = logits.max(1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

                # Update the progress bar
                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

            # Calculate average loss and accuracy
            avg_train_loss = running_loss / len(train_loader)
            train_accuracy = correct_predictions / total_samples
            avg_val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)

            performance.append({
                "avg_train_loss": avg_train_loss,
                "train_accuracy": train_accuracy,
                "avg_val_loss": avg_val_loss,
                "val_accuracy": val_accuracy
            })
            print(
                f"Train Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f} "
                f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}"
            )

            # Check for early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                epochs_without_improvement = 0  # Reset counter if there's an improvement

                # Save the model checkpoint for the best model
                best_model_state = {
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                }
            else:
                epochs_without_improvement += 1

            # Early stopping condition
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break  # Stop training if no improvement for 'patience' epochs

    # Save the performance list to a JSON file
    with open("performance.json", "w") as f:
        json.dump(performance, f, indent=4)
    torch.save(best_model_state, 'model.ckpt')


def test(model, test_loader, device):
    """
    Get predictions for the test set.

    Args:
        model (CNN): classifier to evaluate.
        test_loader (torch.utils.data.DataLoader): Data loader for the test set.
        device (torch.device): Device to use for evaluation.

    Returns:
        float: Average loss on the test set.
        float: Accuracy on the test set.
    """
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    with torch.no_grad():
        all_preds = []

        for inputs, labels in test_loader:
            # Move inputs and labels to device
            inputs = inputs.to(device)

            logits = model(inputs)

            _, predictions = torch.max(logits, dim=1)
            preds = list(zip(labels, predictions.tolist()))
            all_preds.extend(preds)

        return all_preds


def write_predictions(preds, filename):
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for im, pred in preds:
            writer.writerow((im, pred))


def main(args):
    image_net_mean = torch.Tensor([0.485, 0.456, 0.406])
    image_net_std = torch.Tensor([0.229, 0.224, 0.225])

    # Define data transformation
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((128, 128)),
        transforms.Normalize(image_net_mean, image_net_std),
    ])

    # Separate transforms for training and validation
    train_transform = create_train_transform()
    val_transform = create_val_transform()

    # Create datasets
    data_root = 'data'
    miniplaces_train = MiniPlaces(data_root,
                                  split='train',
                                  transform=data_transform)
    miniplaces_val = MiniPlaces(data_root,
                                split='val',
                                transform=data_transform,
                                label_dict=miniplaces_train.label_dict)

    # Create the dataloaders

    # Define the batch size and number of workers
    batch_size = int(args.batch_size)
    num_workers = 2

    # Create DataLoader for training and validation sets
    train_loader = DataLoader(miniplaces_train,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True)
    val_loader = DataLoader(miniplaces_val,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=False)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu')  # TODO: check cuda

    model = MyModel(num_classes=len(miniplaces_train.label_dict))

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4, amsgrad=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, dampening=0, weight_decay=1e-4, nesterov=True)

    print("PARAMS NUM:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    criterion = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)

    if not args.test:
        train(model, train_loader, val_loader, optimizer, criterion,
              device, num_epochs=int(args.epochs))

    else:
        miniplaces_test = MiniPlaces(data_root,
                                     split='test',
                                     transform=data_transform)
        test_loader = DataLoader(miniplaces_test,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False)
        checkpoint = torch.load(args.checkpoint, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        preds = test(model, test_loader, device)
        write_predictions(preds, 'predictions.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--gpu', default=0)
    parser.add_argument('--epochs', default=100)
    parser.add_argument('--batch_size', default=32)
    args = parser.parse_args()
    main(args)
