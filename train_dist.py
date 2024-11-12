#!/usr/bin/env python3
import os
import csv
import json
import warnings
from tqdm import tqdm
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.distributed import DistributedSampler
import argparse
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from model import MyModel


def setup(rank, world_size, port):
    """
    Initialize the distributed training environment.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes (GPUs).
        port (int): The port number for communication.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    """
    Clean up distributed training environment
    """
    if dist.is_initialized():
        dist.barrier()  # Synchronize all processes before destroying process group
        dist.destroy_process_group()
        torch.cuda.synchronize()


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
    model.eval()

    with torch.no_grad():
        total_loss = 0.0
        num_correct = 0
        num_correct_top5 = 0
        num_samples = 0

        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            _, predictions = torch.max(logits, dim=1)
            num_correct += (predictions == labels).sum().item()

            _, top5_predictions = torch.topk(logits, 5, dim=1)
            num_correct_top5 += (top5_predictions == labels.unsqueeze(1)).any(dim=1).sum().item()

            num_samples += len(inputs)

        # Gather metrics from all processes
        world_size = dist.get_world_size()
        total_loss = torch.tensor(total_loss).to(device)
        num_correct = torch.tensor(num_correct).to(device)
        num_correct_top5 = torch.tensor(num_correct_top5).to(device)
        num_samples = torch.tensor(num_samples).to(device)

        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct_top5, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)

        avg_loss = (total_loss / world_size).item() / len(test_loader)
        accuracy = (num_correct / num_samples).item()
        top5_accuracy = (num_correct_top5 / num_samples).item()

    return avg_loss, accuracy, top5_accuracy


def train_worker(rank, world_size, args):
    """
    Train the model in a distributed setup.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes (GPUs).
        args (argparse.Namespace): Command-line arguments.
    """
    try:
        warnings.filterwarnings("ignore")
        setup(rank, world_size, args.port)
        device = torch.device(f'cuda:{rank}')

        # Define early stopping parameters
        patience = 10  # Number of epochs to wait for improvement
        best_val_accuracy = 0.0  # Best validation accuracy so far
        epochs_without_improvement = 0  # Counter for epochs without improvement
        best_model_state = None  # To store the state of the best model

        last_lr = 0

        # Separate transforms for training and validation
        train_transform = create_train_transform()
        val_transform = create_val_transform()

        # Create datasets
        data_root = 'data'
        miniplaces_train = MiniPlaces(data_root, split='train', transform=train_transform)
        miniplaces_val = MiniPlaces(data_root, split='val', transform=val_transform,
                                    label_dict=miniplaces_train.label_dict)

        # Create distributed samplers
        train_sampler = DistributedSampler(miniplaces_train, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(miniplaces_val, num_replicas=world_size, rank=rank)

        # Create dataloaders
        train_loader = DataLoader(miniplaces_train, batch_size=args.batch_size,
                                  num_workers=2, sampler=train_sampler,
                                  pin_memory=True)
        val_loader = DataLoader(miniplaces_val, batch_size=args.batch_size,
                                num_workers=2, sampler=val_sampler,
                                pin_memory=True)

        # Create model and move to GPU
        model = MyModel(num_classes=len(miniplaces_train.label_dict), dropout_rate=0.2)
        model = model.to(device)
        model = DDP(model, device_ids=[rank])

        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                                    dampening=0, weight_decay=1e-4, nesterov=True)
        criterion = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)

        if args.checkpoint:
            map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
            checkpoint = torch.load(args.checkpoint, map_location=map_location)
            model.module.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Initialize the ReduceLROnPlateau scheduler
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=4)

        if not args.test:
            # Training loop
            performance = []
            for epoch in range(args.epochs):
                model.train()
                train_sampler.set_epoch(epoch)  # Important for proper shuffling

                running_loss = 0.0
                correct_predictions = 0
                total_samples = 0

                if rank == 0:  # Only show progress bar on rank 0
                    pbar = tqdm(total=len(train_loader),
                                desc=f'Epoch {epoch + 1}/{args.epochs}',
                                position=0, leave=True)

                for inputs, labels in train_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    logits = model(inputs)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss += loss.item()
                    _, predicted = logits.max(1)
                    correct_predictions += (predicted == labels).sum().item()
                    total_samples += labels.size(0)

                    if rank == 0:
                        pbar.update(1)
                        pbar.set_postfix(loss=loss.item())

                if rank == 0:
                    pbar.close()

                # Evaluate and log metrics
                avg_train_loss = running_loss / len(train_loader)
                train_accuracy = correct_predictions / total_samples
                avg_val_loss, val_accuracy, val_top5_accuracy = evaluate(model, val_loader, criterion, device)

                # Step the scheduler with the validation loss
                scheduler.step(avg_val_loss)
                if scheduler.get_last_lr()[0] != last_lr:
                    last_lr = scheduler.get_last_lr()[0]
                    if epoch != 0:
                        print(f"New learning rate: {scheduler.get_last_lr()[0]}")

                if rank == 0:  # Only save metrics on rank 0
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

            if rank == 0:  # Save performance and the best model checkpoint only on rank 0
                with open("performance.json", "w") as f:
                    json.dump(performance, f, indent=4)
                torch.save(best_model_state, 'model.ckpt')

        else:  # Testing mode
            avg_val_loss, val_accuracy, val_top5_accuracy = evaluate(model, val_loader, criterion, device)
            if rank == 0:
                print(f"\nValidation Loss: {avg_val_loss:.4f}\n"
                      f"Validation Accuracy: {val_accuracy:.4f}\n"
                      f"Validation Top-5 Accuracy: {val_top5_accuracy:.4f}\n")

            miniplaces_test = MiniPlaces(data_root, split='test', transform=val_transform)
            test_loader = DataLoader(miniplaces_test, batch_size=args.batch_size, num_workers=2, shuffle=False)
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.module.load_state_dict(checkpoint['model_state_dict'])

            preds = test(model, test_loader, device)
            if rank == 0:  # Only write predictions on rank 0
                write_predictions(preds, 'predictions.csv')
                print("Predictions saved to predictions.csv\n")

    finally:
        cleanup()
        # Explicit synchronization before exiting
        torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()


def test(model, test_loader, device):
    """
    Test the model on a dataset and return predictions.

    Args:
        model (torch.nn.Module): The model to test.
        test_loader (DataLoader): The DataLoader for the test dataset.
        device (torch.device): The device to run the test on.

    Returns:
        list: A list of (label, prediction) tuples for each image.
    """
    model.eval()
    with torch.no_grad():
        all_preds = []
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            _, predictions = torch.max(logits, dim=1)
            preds = list(zip(labels, predictions.tolist()))
            all_preds.extend(preds)
        return all_preds


def write_predictions(preds, filename):
    """
    Write model predictions to a CSV file.

    Args:
        preds (list): A list of (label, prediction) tuples.
        filename (str): The name of the CSV file to save predictions to.
    """
    with open(filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        for im, pred in preds:
            writer.writerow((im, pred))


def main(args):
    """
    Main function to start the training process using multiple GPUs.

    Args:
        args (argparse.Namespace): Command-line arguments.
    """
    world_size = torch.cuda.device_count()
    try:
        mp.spawn(train_worker,
                 args=(world_size, args),
                 nprocs=world_size,
                 join=True)
    finally:
        # Force cleanup of any remaining CUDA resources
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--checkpoint')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--port', type=int, default=4224)
    args = parser.parse_args()
    main(args)
