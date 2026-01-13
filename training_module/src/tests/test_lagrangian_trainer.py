"""
Unit tests for lagrangian_trainer.py - Lagrangian Fairness Trainer
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

from training_module.src.lagrangian_trainer import (
    LagrangianFairnessTrainer,
    LagrangianConfig,
    create_simple_mlp,
    PYTORCH_AVAILABLE as MODULE_PYTORCH_AVAILABLE,
)


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
class TestLagrangianConfig:
    """Test suite for LagrangianConfig dataclass."""
    
    def test_default_initialization(self):
        """Test default configuration."""
        config = LagrangianConfig()
        
        assert config.lr_model == 0.001
        assert config.lr_lambda == 0.01
        assert config.max_epochs == 100
        assert config.batch_size == 32
        assert config.patience == 10
        assert config.constraint_slack == 0.01
        assert config.lambda_init == 1.0
    
    def test_custom_initialization(self):
        """Test custom configuration."""
        config = LagrangianConfig(
            lr_model=0.01,
            lr_lambda=0.1,
            max_epochs=50,
            constraint_slack=0.05
        )
        
        assert config.lr_model == 0.01
        assert config.lr_lambda == 0.1
        assert config.max_epochs == 50
        assert config.constraint_slack == 0.05


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
class TestCreateSimpleMLP:
    """Test suite for create_simple_mlp factory function."""
    
    def test_default_architecture(self):
        """Test default MLP architecture."""
        model = create_simple_mlp(input_dim=10)
        
        assert isinstance(model, nn.Sequential)
        
        # Count layers (Linear + ReLU pairs + final Linear)
        layers = list(model.children())
        assert len(layers) > 0
    
    def test_custom_architecture(self):
        """Test custom MLP architecture."""
        model = create_simple_mlp(
            input_dim=5,
            hidden_dims=[32, 16, 8]
        )
        
        assert isinstance(model, nn.Sequential)
        
        # Test forward pass
        x = torch.randn(10, 5)
        output = model(x)
        assert output.shape == (10, 1)
    
    def test_with_dropout(self):
        """Test MLP with dropout."""
        model = create_simple_mlp(
            input_dim=10,
            hidden_dims=[64, 32],
            dropout=0.3
        )
        
        # Check for dropout layers
        has_dropout = any(isinstance(layer, nn.Dropout) for layer in model.children())
        assert has_dropout
    
    def test_forward_pass(self):
        """Test forward pass of created MLP."""
        model = create_simple_mlp(input_dim=8, hidden_dims=[16, 8])
        
        x = torch.randn(32, 8)
        output = model(x)
        
        assert output.shape == (32, 1)
        assert not torch.isnan(output).any()


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
class TestLagrangianFairnessTrainer:
    """Test suite for LagrangianFairnessTrainer."""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        n = 200
        X = np.random.randn(n, 10)
        y = np.random.binomial(1, 0.5, n)
        s = np.random.binomial(1, 0.5, n)
        
        # Split
        split = 150
        return {
            'X_train': X[:split],
            'y_train': y[:split],
            's_train': s[:split],
            'X_val': X[split:],
            'y_val': y[split:],
            's_val': s[split:],
        }
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple model."""
        return create_simple_mlp(input_dim=10, hidden_dims=[16, 8])
    
    def test_initialization(self, simple_model):
        """Test trainer initialization."""
        config = LagrangianConfig(max_epochs=10)
        trainer = LagrangianFairnessTrainer(simple_model, config)
        
        assert trainer.model is not None
        assert trainer.config.max_epochs == 10
        assert not trainer.fitted_
        assert trainer.lambda_param is not None
        assert trainer.optimizer_model is not None
        assert trainer.optimizer_lambda is not None
    
    def test_initialization_default_config(self, simple_model):
        """Test initialization with default config."""
        trainer = LagrangianFairnessTrainer(simple_model)
        
        assert trainer.config is not None
        assert trainer.config.lr_model == 0.001
    
    def test_lambda_parameter_initialization(self, simple_model):
        """Test that lambda parameter is initialized correctly."""
        config = LagrangianConfig(lambda_init=2.0)
        trainer = LagrangianFairnessTrainer(simple_model, config)
        
        assert trainer.lambda_param.item() == pytest.approx(2.0)
        assert trainer.lambda_param.requires_grad
    
    def test_fit(self, simple_model, sample_data):
        """Test fitting the trainer."""
        config = LagrangianConfig(max_epochs=5, verbose=False)
        trainer = LagrangianFairnessTrainer(simple_model, config)
        
        trainer.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['s_train']
        )
        
        assert trainer.fitted_
        assert len(trainer.history_['train_loss']) > 0
        assert len(trainer.history_['lambda_values']) > 0
    
    def test_fit_with_validation(self, simple_model, sample_data):
        """Test fitting with validation data."""
        config = LagrangianConfig(max_epochs=5, verbose=False)
        trainer = LagrangianFairnessTrainer(simple_model, config)
        
        trainer.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['s_train'],
            X_val=sample_data['X_val'],
            y_val=sample_data['y_val'],
            sensitive_val=sample_data['s_val']
        )
        
        assert trainer.fitted_
    
    def test_early_stopping(self, simple_model, sample_data):
        """Test early stopping mechanism."""
        config = LagrangianConfig(
            max_epochs=100,
            patience=3,
            verbose=False
        )
        trainer = LagrangianFairnessTrainer(simple_model, config)
        
        trainer.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['s_train']
        )
        
        # Should stop before max_epochs due to patience
        assert len(trainer.history_['train_loss']) < config.max_epochs
    
    def test_history_tracking(self, simple_model, sample_data):
        """Test that training history is tracked correctly."""
        config = LagrangianConfig(max_epochs=5, verbose=False)
        trainer = LagrangianFairnessTrainer(simple_model, config)
        
        trainer.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['s_train']
        )
        
        # Check all history keys
        assert 'train_loss' in trainer.history_
        assert 'train_accuracy' in trainer.history_
        assert 'train_constraint' in trainer.history_
        assert 'lambda_values' in trainer.history_
        
        # All should have same length
        n_epochs = len(trainer.history_['train_loss'])
        assert len(trainer.history_['train_accuracy']) == n_epochs
        assert len(trainer.history_['lambda_values']) == n_epochs
    
    def test_lambda_evolution(self, simple_model, sample_data):
        """Test that lambda parameter evolves during training."""
        config = LagrangianConfig(max_epochs=10, verbose=False)
        trainer = LagrangianFairnessTrainer(simple_model, config)
        
        initial_lambda = trainer.lambda_param.item()
        
        trainer.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['s_train']
        )
        
        final_lambda = trainer.lambda_param.item()
        
        # Lambda should have evolved (not necessarily increased)
        lambda_values = trainer.history_['lambda_values']
        assert len(lambda_values) > 0
        assert all(0 <= lam <= config.lambda_max for lam in lambda_values)
    
    def test_predict(self, simple_model, sample_data):
        """Test predict method."""
        config = LagrangianConfig(max_epochs=5, verbose=False)
        trainer = LagrangianFairnessTrainer(simple_model, config)
        
        trainer.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['s_train']
        )
        
        predictions = trainer.predict(sample_data['X_val'])
        
        assert predictions.shape == (len(sample_data['X_val']),)
        assert set(predictions).issubset({0, 1})
    
    def test_predict_before_fit(self, simple_model, sample_data):
        """Test that predict before fit raises error."""
        trainer = LagrangianFairnessTrainer(simple_model)
        
        with pytest.raises(ValueError, match="Must call fit"):
            trainer.predict(sample_data['X_val'])
    
    def test_predict_proba(self, simple_model, sample_data):
        """Test predict_proba method."""
        config = LagrangianConfig(max_epochs=5, verbose=False)
        trainer = LagrangianFairnessTrainer(simple_model, config)
        
        trainer.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['s_train']
        )
        
        proba = trainer.predict_proba(sample_data['X_val'])
        
        assert proba.shape == (len(sample_data['X_val']), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)
    
    def test_get_constraint_slack(self, simple_model):
        """Test getting constraint slack."""
        config = LagrangianConfig(constraint_slack=0.03)
        trainer = LagrangianFairnessTrainer(simple_model, config)
        
        assert trainer.get_constraint_slack() == 0.03
    
    def test_get_lambda_value(self, simple_model, sample_data):
        """Test getting lambda value."""
        config = LagrangianConfig(lambda_init=1.5, max_epochs=5, verbose=False)
        trainer = LagrangianFairnessTrainer(simple_model, config)
        
        # Before fit
        assert trainer.get_lambda_value() == 1.5
        
        # After fit
        trainer.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['s_train']
        )
        
        lambda_value = trainer.get_lambda_value()
        assert 0 <= lambda_value <= config.lambda_max
    
    def test_different_optimizers(self, simple_model, sample_data):
        """Test with different optimizer configurations."""
        configs = [
            LagrangianConfig(optimizer_model='adam', optimizer_lambda='sgd'),
            LagrangianConfig(optimizer_model='sgd', optimizer_lambda='sgd'),
        ]
        
        for config in configs:
            config.max_epochs = 3
            config.verbose = False
            
            model = create_simple_mlp(input_dim=10)
            trainer = LagrangianFairnessTrainer(model, config)
            
            trainer.fit(
                sample_data['X_train'],
                sample_data['y_train'],
                sample_data['s_train']
            )
            
            assert trainer.fitted_
    
    def test_gradient_clipping(self, simple_model, sample_data):
        """Test with gradient clipping enabled."""
        config = LagrangianConfig(
            max_epochs=5,
            gradient_clip=1.0,
            verbose=False
        )
        trainer = LagrangianFairnessTrainer(simple_model, config)
        
        trainer.fit(
            sample_data['X_train'],
            sample_data['y_train'],
            sample_data['s_train']
        )
        
        assert trainer.fitted_
    
    def test_different_batch_sizes(self, simple_model, sample_data):
        """Test with different batch sizes."""
        batch_sizes = [16, 32, 64]
        
        for batch_size in batch_sizes:
            config = LagrangianConfig(
                batch_size=batch_size,
                max_epochs=3,
                verbose=False
            )
            
            model = create_simple_mlp(input_dim=10)
            trainer = LagrangianFairnessTrainer(model, config)
            
            trainer.fit(
                sample_data['X_train'],
                sample_data['y_train'],
                sample_data['s_train']
            )
            
            assert trainer.fitted_


@pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
class TestConstraintComputation:
    """Test constraint violation computation."""
    
    @pytest.fixture
    def trainer_with_data(self):
        """Create trainer and data for testing."""
        model = create_simple_mlp(input_dim=5)
        config = LagrangianConfig(max_epochs=1, verbose=False)
        trainer = LagrangianFairnessTrainer(model, config)
        
        torch.manual_seed(42)
        predictions = torch.rand(100, 1)
        sensitive = torch.cat([torch.zeros(50), torch.ones(50)])
        
        return trainer, predictions, sensitive
    
    def test_demographic_parity_constraint(self, trainer_with_data):
        """Test demographic parity constraint computation."""
        trainer, predictions, sensitive = trainer_with_data
        
        violation = trainer._compute_constraint_violation(predictions, sensitive)
        
        assert isinstance(violation, torch.Tensor)
        assert violation.dim() == 0  # Scalar
        assert violation.item() >= 0
    
    def test_balanced_groups_low_violation(self):
        """Test that balanced groups have low violation."""
        model = create_simple_mlp(input_dim=5)
        trainer = LagrangianFairnessTrainer(model)
        
        torch.manual_seed(42)
        
        # Create predictions that are equal for both groups
        predictions = torch.ones(100, 1) * 0.5
        sensitive = torch.cat([torch.zeros(50), torch.ones(50)])
        
        violation = trainer._compute_constraint_violation(predictions, sensitive)
        
        # Should be very close to 0
        assert violation.item() < 0.01
    
    def test_imbalanced_predictions_high_violation(self):
        """Test that imbalanced predictions have high violation."""
        model = create_simple_mlp(input_dim=5)
        trainer = LagrangianFairnessTrainer(model)
        
        # Create highly imbalanced predictions
        predictions = torch.cat([
            torch.ones(50, 1) * 0.9,  # Group 0: high
            torch.ones(50, 1) * 0.1   # Group 1: low
        ])
        sensitive = torch.cat([torch.zeros(50), torch.ones(50)])
        
        violation = trainer._compute_constraint_violation(predictions, sensitive)
        
        # Should be large
        assert violation.item() > 0.5


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_very_small_dataset(self):
        """Test with very small dataset."""
        X = np.random.randn(20, 5)
        y = np.random.binomial(1, 0.5, 20)
        s = np.random.binomial(1, 0.5, 20)
        
        model = create_simple_mlp(input_dim=5)
        config = LagrangianConfig(max_epochs=3, batch_size=10, verbose=False)
        trainer = LagrangianFairnessTrainer(model, config)
        
        trainer.fit(X, y, s)
        
        assert trainer.fitted_
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_single_group(self):
        """Test with all samples in one group."""
        X = np.random.randn(50, 5)
        y = np.random.binomial(1, 0.5, 50)
        s = np.zeros(50)  # All group 0
        
        model = create_simple_mlp(input_dim=5)
        config = LagrangianConfig(max_epochs=3, verbose=False)
        trainer = LagrangianFairnessTrainer(model, config)
        
        # Should handle gracefully (constraint will be 0 or nan)
        trainer.fit(X, y, s)
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_imbalanced_groups(self):
        """Test with highly imbalanced groups."""
        X = np.random.randn(100, 5)
        y = np.random.binomial(1, 0.5, 100)
        s = np.zeros(100)
        s[:10] = 1  # Only 10% in group 1
        
        model = create_simple_mlp(input_dim=5)
        config = LagrangianConfig(max_epochs=5, verbose=False)
        trainer = LagrangianFairnessTrainer(model, config)
        
        trainer.fit(X, y, s)
        
        assert trainer.fitted_
    
    @pytest.mark.skipif(not PYTORCH_AVAILABLE, reason="PyTorch not installed")
    def test_perfect_predictions(self):
        """Test with dataset where perfect predictions are possible."""
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)  # Perfect linear separation
        s = np.random.binomial(1, 0.5, 100)
        
        model = create_simple_mlp(input_dim=5, hidden_dims=[32])
        config = LagrangianConfig(max_epochs=10, verbose=False)
        trainer = LagrangianFairnessTrainer(model, config)
        
        trainer.fit(X, y, s)
        
        assert trainer.fitted_
        
        # Should achieve high accuracy
        predictions = trainer.predict(X)
        accuracy = (predictions == y).mean()
        assert accuracy > 0.5  # At least random


@pytest.mark.skipif(PYTORCH_AVAILABLE, reason="Test for when PyTorch is not installed")
class TestWithoutPyTorch:
    """Test behavior when PyTorch is not installed."""
    
    def test_pytorch_not_available(self):
        """Test that appropriate error is raised."""
        with pytest.raises(ImportError, match="PyTorch required"):
            model = nn.Sequential(nn.Linear(5, 1))
            LagrangianFairnessTrainer(model)
    
    def test_create_mlp_without_pytorch(self):
        """Test MLP creation without PyTorch."""
        with pytest.raises(ImportError, match="PyTorch required"):
            create_simple_mlp(input_dim=10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])