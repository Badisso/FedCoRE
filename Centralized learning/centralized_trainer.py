import torch
import torch.nn as nn
# import wandb
import logging


class CentralizedTrainer(object):
    def __init__(self, dataset, model, device, args):
        self.device = device
        self.args = args

        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset

        self.train_global = train_data_global
        self.test_global = test_data_global
        self.model = model.to(self.device)

        # Classification loss
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        if self.args.client_optimizer == "sgd":
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.args.lr,
                momentum=getattr(self.args, "momentum", 0.9)
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                weight_decay=self.args.wd
            )

        # Best accuracy tracker
        self.best_test_acc = 0.0

    def train(self):
        for epoch in range(self.args.epochs):
            self.train_one_epoch(epoch)
            self.eval_and_log(epoch)

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        correct, total, running_loss = 0, 0, 0.0

        for x, labels in self.train_global:
            x = x.view(x.size(0), -1).to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100.0 * correct / total
        avg_loss = running_loss / len(self.train_global)

        logging.info(f"[Epoch {epoch_idx}] Train Loss={avg_loss:.4f}, Accuracy={acc:.2f}%")

    def eval_and_log(self, epoch_idx):
        # Train metrics
        train_loss, train_acc = self.compute_metrics(self.train_global)
        # Test metrics
        test_loss, test_acc = self.compute_metrics(self.test_global)

        logging.info(f"[Epoch {epoch_idx}] Train Accuracy={train_acc:.2f}%, Loss={train_loss:.4f}")
        logging.info(f"[Epoch {epoch_idx}] Test Accuracy={test_acc:.2f}%, Loss={test_loss:.4f}")


        # Save best model
        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc
            torch.save(self.model.state_dict(), "best_model.pth")
            logging.info(f"New best model saved with Test Accuracy={test_acc:.2f}%")

    def compute_metrics(self, dataloader):
        """Compute average loss and accuracy for a given dataloader."""
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for x, labels in dataloader:
                x = x.view(x.size(0), -1).to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(x)
                loss_sum += self.criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = loss_sum / len(dataloader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy
