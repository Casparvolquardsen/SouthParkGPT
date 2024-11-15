import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate import evaluate_loss_acc, evaluate_top_k_accuracy
from tests.test_config import get_test_config


@pytest.fixture
def set_up():
    # Initialize the model, criterion, and device
    device = torch.device('cpu')
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 60))
    criterion = nn.CrossEntropyLoss()

    # Create dummy data
    src_data = torch.randn(100, 10)  # 100 samples, 10 features each
    tgt_data = torch.randint(0, 10, (100,))  # 100 labels for classification

    dataset = TensorDataset(src_data, tgt_data)
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False)

    config = get_test_config()
    config.vocab_size = 60

    return model, criterion, device, data_loader, config


def test_evaluate_loss_range(set_up):
    model, criterion, device, data_loader, config = set_up

    loss, accuracy, _ = evaluate_loss_acc(model, data_loader, criterion, config)
    # Check that loss is non-negative
    assert loss >= 0, "Loss should be non-negative"


def test_evaluate_accuracy_range(set_up):
    model, criterion, device, data_loader, config = set_up

    loss, accuracy, _ = evaluate_loss_acc(model, data_loader, criterion, config)
    # Check that accuracy is between 0 and 100
    assert accuracy >= 0, "Accuracy should be at least 0"
    assert accuracy <= 100, "Accuracy should be at most 100"


def test_evaluate_accuracy_dtype(set_up):
    model, criterion, device, data_loader, config = set_up

    _, accuracy, _ = evaluate_loss_acc(model, data_loader, criterion, config)
    # Check that accuracy is a float
    assert isinstance(accuracy, float), "Accuracy should be a float"


def test_evaluate_top_k_accuracy(set_up):
    probabilities = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1], [0.25, 0.24, 0.26, 0.25]]
        ## true, false, true = 2/3 = 66%
        # In cases where two or more labels are assigned
        # equal predicted scores, the labels with the highest indices will be chosen first.
    )
    _, _, _, _, config = set_up
    target = torch.tensor([2, 2, 2])
    config.vocab_size = 4
    k = 2

    top_k_accuracy = evaluate_top_k_accuracy(probabilities, target, k)

    assert top_k_accuracy == pytest.approx(66.67, abs=1e-2)
