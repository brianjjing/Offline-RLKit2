#!/usr/bin/env python3
"""
Example usage of load_dataset_with_validation_split function.
This shows how to integrate the function into existing training scripts.
"""

import argparse
import numpy as np
import os
import sys

# Add parent directory to path to import from common
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.util import load_dataset_with_validation_split


def example_train_integration(args, env):
    """
    Example showing how to replace existing data loading logic.
    This is how you would modify the train() function in train.py or mbpo_kde/train.py
    """

    print("=" * 60)
    print("Example: Integrating into existing training script")
    print("=" * 60)

    # OLD WAY (commented out):
    # if args.data_path != None:
    #     try:
    #         with open(args.data_path, "rb") as f:
    #             dataset = pickle.load(f)
    #             print('opened the pickle file for synthetic dataset')
    #     except:
    #         dataset = np.load(args.data_path)
    #         dataset = {k: dataset[k] for k in dataset.files}
    #         print('opened the npz file for synthetic dataset')
    #     dataset = {k: v[:1000] for k, v in dataset.items()}
    #     buffer_len = len(dataset['observations'])
    # else:
    #     if args.task == "abiomed":
    #         dataset1 = env.world_model.data_train
    #         dataset2 = env.world_model.data_val
    #         dataset3 = env.world_model.data_test
    #         dataset = [dataset1, dataset2, dataset3]
    #         buffer_len = len(dataset1.data) + len(dataset2.data) + len(dataset3.data)
    #     else:
    #         dataset = env.get_dataset()

    # NEW WAY (using the function):
    dataset_result = load_dataset_with_validation_split(
        args=args,
        env=env,
        val_split_ratio=0.2,  # 20% for validation
        max_samples=getattr(args, 'max_samples', None),  # Optional sample limit
        required_keys=['observations', 'actions', 'rewards', 'terminals']
    )

    # Extract the components
    train_data = dataset_result['train_data']
    val_data = dataset_result['val_data']
    test_data = dataset_result['test_data']  # May be None
    buffer_len = dataset_result['buffer_len']
    data_info = dataset_result['data_info']

    print(f"✓ Loaded dataset from {data_info['source']}")
    print(f"✓ Buffer length: {buffer_len}")
    print(f"✓ Train/val split created: {data_info['created_val_split']}")

    # Now you can use train_data and val_data in your training loop
    # For example:
    if data_info['source'] == 'abiomed_env':
        # Abiomed data has different structure (.data attribute)
        print(f"✓ Train samples: {len(train_data.data)}")
        print(f"✓ Val samples: {len(val_data.data)}")
        if test_data:
            print(f"✓ Test samples: {len(test_data.data)}")
    else:
        # Standard dictionary format
        print(f"✓ Train samples: {len(train_data['observations'])}")
        print(f"✓ Val samples: {len(val_data['observations'])}")

    return train_data, val_data, test_data, buffer_len


def example_density_model_training():
    """
    Example showing how to use the function specifically for training density models.
    This could be used with the RealNVP model we created earlier.
    """

    print("\n" + "=" * 60)
    print("Example: Using for density model training")
    print("=" * 60)

    # Simulate command line arguments for density model training
    args = argparse.Namespace(
        data_path=None,  # Use environment dataset
        task='halfcheetah-medium-v2',
        max_samples=5000  # Limit for faster training
    )

    # Mock environment for this example
    class MockD4RLEnv:
        def get_dataset(self):
            import numpy as np
            return {
                'observations': np.random.randn(10000, 17),
                'actions': np.random.randn(10000, 6),
                'rewards': np.random.randn(10000),
                'terminals': np.random.choice([0, 1], 10000).astype(bool),
                'next_observations': np.random.randn(10000, 17)
            }

    env = MockD4RLEnv()

    # Load dataset with validation split
    dataset_result = load_dataset_with_validation_split(
        args=args,
        env=env,
        val_split_ratio=0.15,  # 15% for validation
        required_keys=['observations', 'actions']  # Only need these for density model
    )

    train_data = dataset_result['train_data']
    val_data = dataset_result['val_data']

    print(f"✓ Ready for density model training:")
    print(f"  - Train observations: {train_data['observations'].shape}")
    print(f"  - Val observations: {val_data['observations'].shape}")
    print(f"  - Train actions: {train_data['actions'].shape}")
    print(f"  - Val actions: {val_data['actions'].shape}")

    # Now you could train a RealNVP model:
    # from models.realnvp import RealNVP
    # import torch
    #
    # # Combine observations and actions for density modeling
    # train_input = torch.FloatTensor(np.concatenate([
    #     train_data['observations'],
    #     train_data['actions']
    # ], axis=1))
    #
    # val_input = torch.FloatTensor(np.concatenate([
    #     val_data['observations'],
    #     val_data['actions']
    # ], axis=1))
    #
    # model = RealNVP(input_dim=train_input.shape[1])
    # history = model.fit(train_input, val_input, epochs=50)

    return train_data, val_data


def example_backwards_compatibility():
    """
    Example showing backwards compatibility with existing code patterns.
    """

    print("\n" + "=" * 60)
    print("Example: Backwards compatibility")
    print("=" * 60)

    # For cases where you need the old format (single dataset dict)
    args = argparse.Namespace(
        data_path=None,
        task='test-env'
    )

    class MockEnv:
        def get_dataset(self):
            import numpy as np
            return {
                'observations': np.random.randn(1000, 10),
                'actions': np.random.randn(1000, 4),
                'rewards': np.random.randn(1000),
                'terminals': np.random.choice([0, 1], 1000).astype(bool)
            }

    env = MockEnv()

    # Load with the new function
    dataset_result = load_dataset_with_validation_split(args=args, env=env)

    # Convert back to old format if needed for legacy code
    if dataset_result['data_info']['source'] != 'abiomed_env':
        # Combine train and val back into single dataset for backwards compatibility
        train_data = dataset_result['train_data']
        val_data = dataset_result['val_data']

        combined_dataset = {}
        for key in train_data.keys():
            combined_dataset[key] = np.concatenate([train_data[key], val_data[key]], axis=0)

        print(f"✓ Combined dataset shape: {combined_dataset['observations'].shape}")
        print("✓ Can still use legacy code that expects single dataset dict")

    # Or keep separate for modern training with validation
    else:
        print("✓ Using separate train/val splits for modern training")


def main():
    """Run example usage demonstrations."""

    print("Dataset Loading Function - Usage Examples")
    print("=" * 70)

    # Example 1: Integration into training script
    args = argparse.Namespace(data_path=None, task='test')

    class MockEnv:
        def get_dataset(self):
            import numpy as np
            return {
                'observations': np.random.randn(2000, 8),
                'actions': np.random.randn(2000, 3),
                'rewards': np.random.randn(2000),
                'terminals': np.random.choice([0, 1], 2000).astype(bool)
            }

    env = MockEnv()

    train_data, val_data, test_data, buffer_len = example_train_integration(args, env)

    # Example 2: Density model training
    example_density_model_training()

    # Example 3: Backwards compatibility
    example_backwards_compatibility()

    print("\n" + "=" * 70)
    print("✅ All examples completed successfully!")
    print("=" * 70)

    print("\n📋 Summary of benefits:")
    print("  ✓ Unified interface for all data sources")
    print("  ✓ Automatic validation split creation")
    print("  ✓ Robust error handling and validation")
    print("  ✓ Backwards compatibility with existing code")
    print("  ✓ Configurable split ratios and sample limits")
    print("  ✓ Detailed logging and metadata")


if __name__ == "__main__":
    main()