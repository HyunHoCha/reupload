from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.optim as optim

from config import CLASSIFICATION_OUTPUT_THRESHOLD


@dataclass
class RegressionResult:
    losses: list[float]
    best_loss: float
    best_epoch: int
    best_predictions: torch.Tensor


@dataclass
class ClassificationResult:
    losses: list[float]
    train_accuracies: list[float]
    test_accuracies: list[float]
    best_test_accuracy: float
    best_epoch: int
    best_test_predictions: torch.Tensor
    best_test_outputs: torch.Tensor


def binary_predictions(outputs: torch.Tensor) -> torch.Tensor:
    return (outputs >= CLASSIFICATION_OUTPUT_THRESHOLD).to(torch.long)


def accuracy(outputs: torch.Tensor, labels: torch.Tensor) -> float:
    preds = binary_predictions(outputs)
    return (preds == labels.to(torch.long)).to(torch.float32).mean().item()


def train_regression(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epochs: int,
    lr: float,
    print_every: int = 100,
) -> RegressionResult:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    best_loss = float("inf")
    best_epoch = 0
    best_predictions = torch.empty_like(y)

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(x).view(-1)
        loss = torch.mean((preds - y) ** 2)
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)
        if loss_value < best_loss:
            best_loss = loss_value
            best_epoch = epoch + 1
            best_predictions = preds.detach().cpu()
        if print_every and ((epoch + 1) % print_every == 0 or epoch == 0):
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss_value:.10g} | Min. Loss: {best_loss:.10g}")

    return RegressionResult(losses, best_loss, best_epoch, best_predictions)


def train_classifier(
    model: torch.nn.Module,
    rho_train: torch.Tensor,
    y_train: torch.Tensor,
    rho_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int,
    lr: float,
    print_every: int = 50,
) -> ClassificationResult:
    optimizer = optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []
    train_accs: list[float] = []
    test_accs: list[float] = []
    best_test_acc = -1.0
    best_epoch = 0
    best_test_predictions = torch.empty_like(y_test, dtype=torch.long).cpu()
    best_test_outputs = torch.empty_like(y_test).cpu()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs_train = model(rho_train).view(-1)
        loss = torch.mean((outputs_train - y_train) ** 2)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            outputs_train_eval = model(rho_train).view(-1)
            outputs_test = model(rho_test).view(-1)
            train_acc = accuracy(outputs_train_eval, y_train)
            test_acc = accuracy(outputs_test, y_test)

        losses.append(loss.item())
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch + 1
            best_test_outputs = outputs_test.detach().cpu()
            best_test_predictions = binary_predictions(outputs_test).detach().cpu()

        if print_every and ((epoch + 1) % print_every == 0 or epoch == 0):
            print(
                f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.10g} | "
                f"Train acc: {train_acc:.4f} | Test acc: {test_acc:.4f} | "
                f"Best test acc: {best_test_acc:.4f}"
            )

    return ClassificationResult(
        losses=losses,
        train_accuracies=train_accs,
        test_accuracies=test_accs,
        best_test_accuracy=best_test_acc,
        best_epoch=best_epoch,
        best_test_predictions=best_test_predictions,
        best_test_outputs=best_test_outputs,
    )
