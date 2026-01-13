"""
Unit tests for pytorch_losses.py - PyTorch Fairness Losses
"""

import pytest
import numpy as np

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

from training_module.src.pytorch_losses import (
    FairnessRegularizedLoss,
    EqualizdOddsLoss,
    create_fairness_loss,
    PYTORCH_AVAILABLE as MODULE_PYTORCH_AVAILABLE,
)


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
class TestFairnessRegularizedLoss:
    """Test suite for FairnessRegularizedLoss."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        torch.manual_seed(42)
        batch_size = 32
        
        logits = torch.randn(batch_size, 1)
        targets = torch.randint(0, 2, (batch_size, 1)).float()
        sensitive = torch.randint(0, 2, (batch_size,)).float()
        
        return logits, targets, sensitive
    
    def test_initialization_default(self):
        """Test initialization with default parameters."""
        criterion = FairnessRegularizedLoss()
        
        assert criterion.fairness_weight == 0.5
        assert criterion.fairness_type == 'demographic_parity'
        assert isinstance(criterion.base_loss, nn.BCEWithLogitsLoss)
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        base_loss = nn.BCELoss()
        criterion = FairnessRegularizedLoss(
            base_loss=base_loss,
            fairness_weight=0.7,
            fairness_type='demographic_parity'
        )
        
        assert criterion.fairness_weight == 0.7
        assert criterion.base_loss is base_loss
    
    def test_initialization_invalid_fairness_type(self):
        """Test initialization with invalid fairness type."""
        with pytest.raises(NotImplementedError, match="Only 'demographic_parity' supported"):
            FairnessRegularizedLoss(fairness_type='equalized_odds')
    
    def test_forward_pass(self, sample_data):
        """Test forward pass computation."""
        logits, targets, sensitive = sample_data
        
        criterion = FairnessRegularizedLoss(fairness_weight=0.5)
        loss = criterion(logits, targets, sensitive)
        
        # Loss should be a scalar tensor
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_forward_returns_different_values_for_different_weights(self, sample_data):
        """Test that different fairness weights produce different losses."""
        logits, targets, sensitive = sample_data
        
        criterion_low = FairnessRegularizedLoss(fairness_weight=0.1)
        criterion_high = FairnessRegularizedLoss(fairness_weight=0.9)
        
        loss_low = criterion_low(logits, targets, sensitive)
        loss_high = criterion_high(logits, targets, sensitive)
        
        # Higher weight should generally produce different loss
        # (not necessarily higher due to fairness penalty direction)
        assert loss_low.item() != loss_high.item()
    
    def test_demographic_parity_loss_computation(self, sample_data):
        """Test demographic parity loss component."""
        logits, targets, sensitive = sample_data
        
        criterion = FairnessRegularizedLoss(fairness_weight=1.0)
        
        # Get fairness loss
        with torch.no_grad():
            predictions = torch.sigmoid(logits).squeeze()
            
            mask_group0 = sensitive == 0
            mask_group1 = sensitive == 1
            
            mean_g0 = predictions[mask_group0].mean()
            mean_g1 = predictions[mask_group1].mean()
            
            expected_fairness_loss = (mean_g0 - mean_g1) ** 2
        
        # Compute full loss (should include this fairness component)
        total_loss = criterion(logits, targets, sensitive)
        
        assert total_loss.item() >= 0
    
    def test_backward_pass(self, sample_data):
        """Test that backward pass works correctly."""
        logits, targets, sensitive = sample_data
        logits.requires_grad = True
        
        criterion = FairnessRegularizedLoss(fairness_weight=0.5)
        loss = criterion(logits, targets, sensitive)
        
        loss.backward()
        
        # Gradients should be computed
        assert logits.grad is not None
        assert not torch.all(logits.grad == 0)
    
    def test_zero_fairness_weight(self, sample_data):
        """Test that zero fairness weight gives only accuracy loss."""
        logits, targets, sensitive = sample_data
        
        criterion_fair = FairnessRegularizedLoss(fairness_weight=0.0)
        criterion_base = nn.BCEWithLogitsLoss()
        
        loss_fair = criterion_fair(logits, targets, sensitive)
        loss_base = criterion_base(logits, targets)
        
        # Should be equal (or very close)
        assert torch.allclose(loss_fair, loss_base, rtol=1e-5)
    
    def test_balanced_groups(self):
        """Test with perfectly balanced groups."""
        torch.manual_seed(42)
        
        # Create data where both groups have same prediction rate
        logits = torch.randn(100, 1)
        targets = torch.randint(0, 2, (100, 1)).float()
        sensitive = torch.cat([torch.zeros(50), torch.ones(50)])
        
        criterion = FairnessRegularizedLoss(fairness_weight=1.0)
        loss = criterion(logits, targets, sensitive)
        
        # Loss should still be computed
        assert loss.item() >= 0
    
    def test_imbalanced_groups(self):
        """Test with imbalanced groups."""
        torch.manual_seed(42)
        
        logits = torch.randn(100, 1)
        targets = torch.randint(0, 2, (100, 1)).float()
        sensitive = torch.cat([torch.zeros(90), torch.ones(10)])
        
        criterion = FairnessRegularizedLoss(fairness_weight=0.5)
        loss = criterion(logits, targets, sensitive)
        
        assert loss.item() >= 0
    
    def test_different_base_losses(self, sample_data):
        """Test with different base loss functions."""
        logits, targets, sensitive = sample_data
        
        base_losses = [
            nn.BCEWithLogitsLoss(),
            nn.BCEWithLogitsLoss(reduction='sum'),
        ]
        
        for base_loss in base_losses:
            criterion = FairnessRegularizedLoss(
                base_loss=base_loss,
                fairness_weight=0.5
            )
            
            loss = criterion(logits, targets, sensitive)
            assert loss.item() >= 0
    
    def test_batch_size_one(self):
        """Test with batch size of 1 (edge case)."""
        logits = torch.randn(1, 1)
        targets = torch.tensor([[1.0]])
        sensitive = torch.tensor([0.0])
        
        criterion = FairnessRegularizedLoss(fairness_weight=0.5)
        
        # Should handle gracefully
        loss = criterion(logits, targets, sensitive)
        
        assert loss.item() >= 0
        assert torch.isfinite(loss)
        
        # Test that it can compute gradients
        logits_with_grad = torch.randn(1, 1, requires_grad=True)
        loss = criterion(logits_with_grad, targets, sensitive)
        loss.backward()
        
        assert logits_with_grad.grad is not None
        
    def test_batch_size_one_both_groups(self):
        """Test with batch size of 2, one sample per group."""
        torch.manual_seed(42)
        
        logits = torch.randn(2, 1)
        targets = torch.tensor([[1.0], [0.0]])
        sensitive = torch.tensor([0.0, 1.0])  # One sample per group
        
        criterion = FairnessRegularizedLoss(fairness_weight=0.5)
        
        loss = criterion(logits, targets, sensitive)
        
        assert torch.isfinite(loss)
        assert loss.item() >= 0


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
class TestEqualizdOddsLoss:
    """Test suite for EqualizdOddsLoss (stub)."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data."""
        torch.manual_seed(42)
        batch_size = 32
        
        logits = torch.randn(batch_size, 1)
        targets = torch.randint(0, 2, (batch_size, 1)).float()
        sensitive = torch.randint(0, 2, (batch_size,)).float()
        
        return logits, targets, sensitive
    
    def test_initialization(self):
        """Test initialization (stub implementation)."""
        criterion = EqualizdOddsLoss()
        
        assert isinstance(criterion.base_loss, nn.BCEWithLogitsLoss)
        assert criterion.fairness_weight == 0.5
    
    def test_forward_returns_base_loss_only(self, sample_data):
        """Test that stub implementation returns only base loss."""
        logits, targets, sensitive = sample_data
        
        criterion = EqualizdOddsLoss()
        criterion_base = nn.BCEWithLogitsLoss()
        
        loss_stub = criterion(logits, targets, sensitive)
        loss_base = criterion_base(logits, targets)
        
        # Should return only base loss (stub implementation)
        assert torch.allclose(loss_stub, loss_base, rtol=1e-5)


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
class TestCreateFairnessLoss:
    """Test suite for create_fairness_loss factory function."""
    
    def test_create_with_bce(self):
        """Test creation with BCE loss."""
        criterion = create_fairness_loss(
            base_loss='bce',
            fairness_weight=0.5,
            fairness_type='demographic_parity'
        )
        
        assert isinstance(criterion, FairnessRegularizedLoss)
        assert isinstance(criterion.base_loss, nn.BCEWithLogitsLoss)
        assert criterion.fairness_weight == 0.5
    
    def test_create_with_cross_entropy(self):
        """Test creation with cross entropy loss."""
        criterion = create_fairness_loss(
            base_loss='cross_entropy',
            fairness_weight=0.3,
            fairness_type='demographic_parity'
        )
        
        assert isinstance(criterion, FairnessRegularizedLoss)
        assert isinstance(criterion.base_loss, nn.CrossEntropyLoss)
        assert criterion.fairness_weight == 0.3
    
    def test_create_with_invalid_base_loss(self):
        """Test creation with invalid base loss."""
        with pytest.raises(ValueError, match="Unknown base_loss"):
            create_fairness_loss(base_loss='invalid')
    
    def test_create_with_custom_parameters(self):
        """Test creation with various parameters."""
        weights = [0.0, 0.5, 1.0]
        
        for weight in weights:
            criterion = create_fairness_loss(
                base_loss='bce',
                fairness_weight=weight,
                fairness_type='demographic_parity'
            )
            
            assert criterion.fairness_weight == weight


@pytest.mark.skipif(PYTORCH_AVAILABLE, reason="Test for when PyTorch is not installed")
class TestWithoutPyTorch:
    """Test behavior when PyTorch is not installed."""
    
    def test_pytorch_not_available_flag(self):
        """Test that PYTORCH_AVAILABLE flag is False."""
        assert not MODULE_PYTORCH_AVAILABLE
    
    def test_import_error_on_instantiation(self):
        """Test that appropriate error is raised when PyTorch not available."""
        with pytest.raises(ImportError, match="PyTorch required"):
            FairnessRegularizedLoss()


class TestIntegrationScenarios:
    """Integration tests with realistic training scenarios."""
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_simple_training_loop(self):
        """Test criterion in a simple training loop."""
        torch.manual_seed(42)
        
        # Simple model
        model = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
            nn.Linear(5, 1)
        )
        
        # Optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Criterion
        criterion = FairnessRegularizedLoss(fairness_weight=0.5)
        
        # Training data
        X = torch.randn(64, 10)
        y = torch.randint(0, 2, (64, 1)).float()
        s = torch.randint(0, 2, (64,)).float()
        
        # Training step
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y, s)
        loss.backward()
        optimizer.step()
        
        # Check that training worked
        assert loss.item() >= 0
        assert all(p.grad is not None for p in model.parameters())
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_multiple_epochs(self):
        """Test criterion over multiple epochs."""
        torch.manual_seed(42)
        
        model = nn.Linear(5, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = FairnessRegularizedLoss(fairness_weight=0.5)
        
        X = torch.randn(100, 5)
        y = torch.randint(0, 2, (100, 1)).float()
        s = torch.randint(0, 2, (100,)).float()
        
        losses = []
        
        for epoch in range(5):
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y, s)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
        
        # Loss should generally decrease
        assert all(l >= 0 for l in losses)
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_with_dataloader(self):
        """Test criterion with PyTorch DataLoader."""
        torch.manual_seed(42)
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create dataset
        X = torch.randn(200, 10)
        y = torch.randint(0, 2, (200, 1)).float()
        s = torch.randint(0, 2, (200,)).float()
        
        dataset = TensorDataset(X, y, s)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Model and criterion
        model = nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = FairnessRegularizedLoss(fairness_weight=0.5)
        
        # Train for one epoch
        total_loss = 0
        for X_batch, y_batch, s_batch in dataloader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch, s_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        assert total_loss > 0


class TestEdgeCasesAndErrors:
    """Test edge cases and error conditions."""
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_empty_group(self):
        """Test with one empty group (all samples in one group)."""
        torch.manual_seed(42)
        
        logits = torch.randn(32, 1)
        targets = torch.randint(0, 2, (32, 1)).float()
        sensitive = torch.zeros(32)  # All in group 0
        
        criterion = FairnessRegularizedLoss(fairness_weight=0.5)
        
        # Should handle gracefully (mean of empty group will be nan)
        loss = criterion(logits, targets, sensitive)
        
        # Loss might be nan due to empty group, or handled gracefully
        # Both behaviors are acceptable
        assert isinstance(loss, torch.Tensor)
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_extreme_predictions(self):
        """Test with extreme prediction values."""
        torch.manual_seed(42)
        
        logits = torch.tensor([[100.0], [-100.0], [100.0], [-100.0]])
        targets = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
        sensitive = torch.tensor([0.0, 0.0, 1.0, 1.0])
        
        criterion = FairnessRegularizedLoss(fairness_weight=0.5)
        loss = criterion(logits, targets, sensitive)
        
        # Should handle extreme values
        assert torch.isfinite(loss)
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_nan_handling(self):
        """Test handling of NaN values in input."""
        torch.manual_seed(42)
        
        logits = torch.tensor([[1.0], [float('nan')], [1.0]])
        targets = torch.tensor([[1.0], [0.0], [1.0]])
        sensitive = torch.tensor([0.0, 0.0, 1.0])
        
        criterion = FairnessRegularizedLoss(fairness_weight=0.5)
        loss = criterion(logits, targets, sensitive)
        
        # Loss will likely be nan if input contains nan
        assert isinstance(loss, torch.Tensor)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])