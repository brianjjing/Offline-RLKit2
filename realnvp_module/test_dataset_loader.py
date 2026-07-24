#!/usr/bin/env python3
"""
Test script for the load_dataset_with_validation_split function.
This demonstrates usage patterns and validates the function works correctly.
"""

import numpy as np
import torch
import pickle
import os
import sys
import tempfile
from argparse import Namespace

# Add parent directory to path to import from common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.util import load_dataset_with_validation_split, validate_dataset_structure


def create_synthetic_dataset(n_samples=1000, obs_dim=10, action_dim=4):
    """Create a synthetic dataset for testing."""
    dataset = {
        'observations': np.random.randn(n_samples, obs_dim).astype(np.float32),
        'actions': np.random.randn(n_samples, action_dim).astype(np.float32),
        'rewards': np.random.randn(n_samples).astype(np.float32),
        'terminals': np.random.choice([0, 1], size=n_samples).astype(bool),
        'next_observations': np.random.randn(n_samples, obs_dim).astype(np.float32)
    }
    return dataset


class MockEnvironment:
    """Mock environment for testing env.get_dataset() functionality."""

    def __init__(self, dataset):
        self.dataset = dataset

    def get_dataset(self):
        return self.dataset


class MockAbiomedEnvironment:
    """Mock Abiomed environment with predefined splits."""

    def __init__(self, train_size=800, val_size=150, test_size=50):
        # Create mock data objects with .data attribute
        class MockData:
            def __init__(self, size):
                self.data = np.random.randn(size, 10)

        class MockWorldModel:
            def __init__(self):
                self.data_train = MockData(train_size)
                self.data_val = MockData(val_size)
                self.data_test = MockData(test_size)

        self.world_model = MockWorldModel()


def test_pickle_loading():
    """Test loading from pickle file."""
    print("=" * 50)
    print("Testing Pickle File Loading")
    print("=" * 50)

    # Create synthetic dataset
    dataset = create_synthetic_dataset(n_samples=1000)

    # Save to temporary pickle file
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        pickle.dump(dataset, f)
        pickle_path = f.name

    try:
        # Create args object
        args = Namespace(data_path=pickle_path, task='test')

        # Load dataset
        result = load_dataset_with_validation_split(
            args=args,
            val_split_ratio=0.2,
            max_samples=500  # Limit for testing
        )

        # Validate results
        print(f"Train data observations shape: {result['train_data']['observations'].shape}")
        print(f"Val data observations shape: {result['val_data']['observations'].shape}")
        print(f"Buffer length: {result['buffer_len']}")
        print(f"Data info: {result['data_info']}")

        # Check split ratio
        total_samples = len(result['train_data']['observations']) + len(result['val_data']['observations'])
        val_ratio = len(result['val_data']['observations']) / total_samples
        print(f"Actual validation ratio: {val_ratio:.3f}")

        assert abs(val_ratio - 0.2) < 0.05, f"Validation split ratio incorrect: {val_ratio}"
        assert result['data_info']['created_val_split'] == True
        assert result['buffer_len'] == 500  # max_samples limit

        print("✓ Pickle loading test passed!")

    finally:
        os.unlink(pickle_path)


def test_npz_loading():
    """Test loading from npz file."""
    print("\n" + "=" * 50)
    print("Testing NPZ File Loading")
    print("=" * 50)

    # Create synthetic dataset
    dataset = create_synthetic_dataset(n_samples=800)

    # Save to temporary npz file
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
        np.savez(f.name, **dataset)
        npz_path = f.name

    try:
        # Create args object
        args = Namespace(data_path=npz_path, task='test')

        # Load dataset
        result = load_dataset_with_validation_split(
            args=args,
            val_split_ratio=0.25
        )

        # Validate results
        print(f"Train data observations shape: {result['train_data']['observations'].shape}")
        print(f"Val data observations shape: {result['val_data']['observations'].shape}")
        print(f"Buffer length: {result['buffer_len']}")

        # Check split ratio
        total_samples = len(result['train_data']['observations']) + len(result['val_data']['observations'])
        val_ratio = len(result['val_data']['observations']) / total_samples
        print(f"Actual validation ratio: {val_ratio:.3f}")

        assert abs(val_ratio - 0.25) < 0.05, f"Validation split ratio incorrect: {val_ratio}"
        assert result['data_info']['format'] == 'npz'

        print("✓ NPZ loading test passed!")

    finally:
        os.unlink(npz_path)


def test_env_dataset_loading():
    """Test loading from environment.get_dataset()."""
    print("\n" + "=" * 50)
    print("Testing Environment Dataset Loading")
    print("=" * 50)

    # Create mock environment
    dataset = create_synthetic_dataset(n_samples=600)
    env = MockEnvironment(dataset)

    # Create args object without data_path
    args = Namespace(task='test-env')

    # Load dataset
    result = load_dataset_with_validation_split(
        args=args,
        env=env,
        val_split_ratio=0.15
    )

    # Validate results
    print(f"Train data observations shape: {result['train_data']['observations'].shape}")
    print(f"Val data observations shape: {result['val_data']['observations'].shape}")
    print(f"Buffer length: {result['buffer_len']}")

    # Check split ratio
    total_samples = len(result['train_data']['observations']) + len(result['val_data']['observations'])
    val_ratio = len(result['val_data']['observations']) / total_samples
    print(f"Actual validation ratio: {val_ratio:.3f}")

    assert abs(val_ratio - 0.15) < 0.05, f"Validation split ratio incorrect: {val_ratio}"
    assert result['data_info']['source'] == 'env_dataset'
    assert result['data_info']['created_val_split'] == True

    print("✓ Environment dataset loading test passed!")


def test_abiomed_loading():
    """Test loading Abiomed dataset with predefined splits."""
    print("\n" + "=" * 50)
    print("Testing Abiomed Dataset Loading")
    print("=" * 50)

    # Create mock Abiomed environment
    env = MockAbiomedEnvironment(train_size=500, val_size=100, test_size=50)

    # Create args object
    args = Namespace(task='abiomed')

    # Load dataset
    result = load_dataset_with_validation_split(
        args=args,
        env=env
    )

    # Validate results
    print(f"Train data size: {len(result['train_data'].data)}")
    print(f"Val data size: {len(result['val_data'].data)}")
    print(f"Test data size: {len(result['test_data'].data)}")
    print(f"Buffer length: {result['buffer_len']}")

    assert len(result['train_data'].data) == 500
    assert len(result['val_data'].data) == 100
    assert len(result['test_data'].data) == 50
    assert result['buffer_len'] == 650
    assert result['data_info']['has_predefined_splits'] == True
    assert result['data_info']['created_val_split'] == False

    print("✓ Abiomed dataset loading test passed!")


def test_dataset_validation():
    """Test dataset structure validation."""
    print("\n" + "=" * 50)
    print("Testing Dataset Validation")
    print("=" * 50)

    # Valid dataset
    valid_dataset = create_synthetic_dataset(n_samples=100)

    try:
        validate_dataset_structure(
            valid_dataset,
            required_keys=['observations', 'actions', 'rewards', 'terminals'],
            min_samples=50
        )
        print("✓ Valid dataset passed validation")
    except ValueError as e:
        print(f"✗ Valid dataset failed validation: {e}")
        raise

    # Test missing keys
    invalid_dataset = {k: v for k, v in valid_dataset.items() if k != 'rewards'}

    try:
        validate_dataset_structure(
            invalid_dataset,
            required_keys=['observations', 'actions', 'rewards', 'terminals']
        )
        print("✗ Invalid dataset should have failed validation")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Missing keys correctly detected")

    # Test inconsistent lengths
    inconsistent_dataset = valid_dataset.copy()
    inconsistent_dataset['actions'] = inconsistent_dataset['actions'][:50]  # Shorter array

    try:
        validate_dataset_structure(
            inconsistent_dataset,
            required_keys=['observations', 'actions', 'rewards', 'terminals']
        )
        print("✗ Inconsistent lengths should have failed validation")
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ Inconsistent lengths correctly detected")


def test_error_handling():
    """Test error handling for invalid inputs."""
    print("\n" + "=" * 50)
    print("Testing Error Handling")
    print("=" * 50)

    # Test missing both data_path and env
    args = Namespace(task='test')

    try:
        load_dataset_with_validation_split(args=args)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly caught error for missing inputs: {e}")

    # Test invalid file path
    args = Namespace(data_path='/nonexistent/file.pkl', task='test')

    try:
        load_dataset_with_validation_split(args=args)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Correctly caught error for invalid file: {e}")


def main():
    """Run all tests."""
    print("Testing load_dataset_with_validation_split function")
    print("=" * 70)

    try:
        test_pickle_loading()
        test_npz_loading()
        test_env_dataset_loading()
        test_abiomed_loading()
        test_dataset_validation()
        test_error_handling()

        print("\n" + "=" * 70)
        print("🎉 All tests passed! The function is working correctly.")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    main()