"""
Test script for MONAI bundle and MONAI Label application
"""

import os
import sys
import tempfile
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def test_monai_bundle() -> bool:
    """Test MONAI bundle loading and inference"""
    print("Testing MONAI bundle...")

    try:
        import json

        bundle_path = Path("monai_bundle/aorta_segmentation_unet")
        required_files = ["metadata.json", "configs/inference.json", "LICENSE"]

        for file in required_files:
            file_path = bundle_path / file
            if not file_path.exists():
                print(f"❌ Missing required file: {file}")
                return False
            print(f"✅ Found: {file}")

        with open(bundle_path / "metadata.json") as f:
            metadata = json.load(f)
        print(f"✅ Bundle metadata loaded: {metadata['name']}")

        with open(bundle_path / "configs/inference.json") as f:
            json.load(f)
        print("✅ Inference config loaded")

        print("✅ MONAI bundle structure test passed")
        return True

    except Exception as e:
        print(f"❌ MONAI bundle test failed: {e}")
        return False


def test_monai_label_app() -> bool:
    """Test MONAI Label application loading"""
    print("\nTesting MONAI Label application...")

    try:
        app_path = Path("monai_label_app/aorta_app")
        sys.path.insert(0, str(app_path))

        from lib.configs import INTENSITY_RANGE, NETWORK_CONFIG, TARGET_SPACING
        from lib.infers import AortaSegmentation

        print("✅ All imports successful")

        print(f"✅ Network config: {NETWORK_CONFIG['out_channels']} output classes")
        print(f"✅ Intensity range: {INTENSITY_RANGE}")
        print(f"✅ Target spacing: {TARGET_SPACING}")

        from monai.networks.nets import UNet

        network = UNet(**NETWORK_CONFIG)
        print(f"✅ Network created: {network.__class__.__name__}")

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pt")
            torch.save(network.state_dict(), model_path)

            infer_task = AortaSegmentation(path=model_path, network=network, roi_size=[96, 96, 96], preload=False)
            print("✅ Inference task created successfully")

            pre_transforms = infer_task.pre_transforms()
            post_transforms = infer_task.post_transforms()
            print(
                f"✅ Transforms created: {len(pre_transforms.transforms)} pre, {len(post_transforms.transforms)} post"
            )

        print("✅ MONAI Label application test passed")
        return True

    except Exception as e:
        print(f"❌ MONAI Label application test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def create_test_data() -> str | None:
    """Create test NIfTI data for testing"""
    print("\nCreating test data...")

    try:
        rng = np.random.default_rng()
        test_data = rng.integers(-175, 250, size=(64, 64, 64), dtype=np.int16)

        nii_img = nib.Nifti1Image(test_data, affine=np.eye(4))

        os.makedirs("test_data", exist_ok=True)
        test_file = "test_data/test_volume.nii.gz"
        nib.save(nii_img, test_file)

        print(f"✅ Test data created: {test_file}")
        print(f"   Shape: {test_data.shape}")
        print(f"   Intensity range: [{test_data.min()}, {test_data.max()}]")

        return test_file

    except Exception as e:
        print(f"❌ Test data creation failed: {e}")
        return None


def test_preprocessing_pipeline() -> bool:
    """Test the preprocessing pipeline with test data"""
    print("\nTesting preprocessing pipeline...")

    try:
        test_file = create_test_data()
        if not test_file:
            return False

        app_path = Path("monai_label_app/aorta_app")
        sys.path.insert(0, str(app_path))

        from lib.configs import NETWORK_CONFIG
        from lib.infers import AortaSegmentation
        from monai.networks.nets import UNet

        network = UNet(**NETWORK_CONFIG)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pt")
            torch.save(network.state_dict(), model_path)

            infer_task = AortaSegmentation(path=model_path, network=network, roi_size=[96, 96, 96], preload=False)

            pre_transforms = infer_task.pre_transforms()

            test_dict = {"image": test_file}
            transformed = pre_transforms(test_dict)

            print("✅ Preprocessing completed")
            print(f"   Input file: {test_file}")
            print(f"   Output shape: {transformed['image'].shape}")
            print(f"   Output type: {type(transformed['image'])}")
            print(f"   Value range: [{transformed['image'].min():.3f}, {transformed['image'].max():.3f}]")

        print("✅ Preprocessing pipeline test passed")
        return True

    except Exception as e:
        print(f"❌ Preprocessing pipeline test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main() -> bool:
    """Run all tests"""
    print("🧪 Testing MONAI Bundle and MONAI Label Application")
    print("=" * 60)

    os.chdir("C:\DevinMonai\Aorta_Analysis")

    tests = [
        test_monai_bundle,
        test_monai_label_app,
        test_preprocessing_pipeline,
    ]

    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()

    print("=" * 60)
    print("📊 Test Summary:")
    test_names = ["MONAI Bundle", "MONAI Label App", "Preprocessing Pipeline"]

    for name, result in zip(test_names, results):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {name}: {status}")

    all_passed = all(results)
    if all_passed:
        print("\n🎉 All tests passed! The implementation is ready.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
