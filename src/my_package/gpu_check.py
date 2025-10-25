#!/usr/bin/env python3
"""GPU availability check using JAX."""

import jax


def main() -> None:
    """Check GPU availability and perform basic operations."""
    print("=" * 50)
    print("GPU AVAILABILITY CHECK")
    print("=" * 50)

    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Local devices: {jax.local_devices()}")

    # Check if GPU is available
    gpu_devices = [d for d in jax.devices() if d.platform in ['gpu', 'cuda']]
    if gpu_devices:
        print(f"\n✅ GPU(s) detected: {len(gpu_devices)} device(s)")
        for i, device in enumerate(gpu_devices):
            print(f"   GPU {i}: {device}")
    else:
        print("\n⚠️  No GPU devices detected - running on CPU")
        print("Remember to unset LD_LIBRARY_PATH.")

    print("\n" + "=" * 50)
    print("GPU availability check completed successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
