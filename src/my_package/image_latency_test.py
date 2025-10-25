#!/usr/bin/env python3
"""Image transfer latency test using Portal.

This script tests the latency of transferring randomly generated images
at different resolutions using the Portal library.
It measures round-trip time, throughput, and provides detailed statistics
for performance analysis.

Usage:
    python image_latency_test.py
        [--port PORT] [--samples SAMPLES] [--warmup WARMUP]
"""

import argparse
import statistics
import time
from typing import Any, Dict, Optional

import numpy as np
import portal


class ImageLatencyTester:
    """Test image transfer latency using Portal."""

    # Test image resolutions (width, height, channels)
    RESOLUTIONS = [
        (64, 64, 3),      # 12 KB
        (128, 128, 3),    # 48 KB
        (256, 256, 3),    # 192 KB
        (512, 512, 3),    # 768 KB
        (1024, 1024, 3),  # 3 MB
        (1920, 1080, 3),  # 6.2 MB (HD)
        (2048, 2048, 3),  # 12 MB
        (4096, 4096, 3),  # 48 MB (4K)
    ]

    def __init__(
            self, port: Optional[int] = None,
            samples: int = 100,
            warmup: int = 10):
        """Initialize the latency tester.

        Args:
            port (int):
                Port number for the Portal server (auto-assigned if None)
            samples (int): Number of samples to collect for each resolution
            warmup (int): Number of warmup rounds before measurement
        """
        self.port = port or portal.free_port()
        self.samples = samples
        self.warmup = warmup
        self.results: Dict[str, Dict[str, Any]] = {}

    def generate_random_image(
            self,
            height: int,
            width: int,
            channels: int) -> np.ndarray:
        """Generate a random image with specified dimensions.

        Args:
            height (int): Image height
            width (int): Image width
            channels (int): Number of color channels

        Returns:
            np.ndarray: Generated random image
        """
        # Generate random uint8 image data
        image = np.random.randint(
            0,
            256,
            size=(
                height,
                width,
                channels),
            dtype=np.uint8)
        return image

    def calculate_image_size(
            self,
            height: int,
            width: int,
            channels: int) -> int:
        """Calculate image size in bytes.

        Args:
            height (int): Image height
            width (int): Image width
            channels (int): Number of color channels

        Returns:
            int: Image size in bytes
        """
        return height * width * channels

    def format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format.

        Args:
            size_bytes (int): Size in bytes

        Returns:
            str: Formatted size string
        """
        size_bytes_f: float = size_bytes
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes_f < 1024.0:
                return f"{size_bytes_f:.1f} {unit}"
            size_bytes_f /= 1024.0
        return f"{size_bytes_f:.1f} TB"

    def server_function(self) -> None:
        """Portal server function that echoes back received images."""
        server = portal.Server(self.port, name='ImageServer')

        def process_image(image_data: np.ndarray) -> np.ndarray:
            """Simple echo function - returns the same image.

            Args:
                image_data (np.ndarray): Received image data

            Returns:
                np.ndarray: Echoed image data
            """
            return image_data

        def get_image_info(image_data: np.ndarray) -> Dict[str, Any]:
            """Returns information about the received image.

            Args:
                image_data (np.ndarray): Received image data

            Returns:
                Dict[str, Any]: Image information
            """
            return {
                'shape': image_data.shape,
                'dtype': str(image_data.dtype),
                'size_bytes': image_data.nbytes,
                'min_val': int(image_data.min()),
                'max_val': int(image_data.max()),
                'mean_val': float(image_data.mean())
            }

        server.bind('process_image', process_image)
        server.bind('get_image_info', get_image_info)

        print(f"🚀 Image server started on port {self.port}")
        server.start(block=True)

    def _perform_warmup(
            self,
            client: portal.Client,
            height: int,
            width: int,
            channels: int) -> bool:
        """Perform warmup rounds to stabilize measurements.

        Args:
            client (portal.Client): Portal client for communication
            height (int): Image height
            width (int): Image width
            channels (int): Number of color channels

        Returns:
            bool: True if warmup successful, False otherwise
        """
        print(f"   🔥 Warming up ({self.warmup} rounds)...")
        for _ in range(self.warmup):
            warmup_image = self.generate_random_image(height, width, channels)
            try:
                client.process_image(warmup_image).result(timeout=30)
            except Exception as e:
                print(f"   ⚠️  Warmup failed: {e}")
                return False
        return True

    def _collect_latency_samples(
        self,
        client: portal.Client,
        height: int,
        width: int,
        channels: int,
        image_size: int
    ) -> tuple[list[float], list[float]]:
        """Collect latency and throughput samples.

        Args:
            client (portal.Client): Portal client for communication
            height (int): Image height
            width (int): Image width
            channels (int): Number of color channels
            image_size (int): Image size in bytes

        Returns:
            tuple[list[float], list[float]]: Latencies and throughputs
        """
        print(f"   📏 Measuring latency ({self.samples} samples)...")
        latencies = []
        throughputs = []

        for i in range(self.samples):
            # Generate random image
            image = self.generate_random_image(height, width, channels)

            # Measure round-trip time
            start_time = time.perf_counter()
            try:
                result = client.process_image(image).result(timeout=30)
                end_time = time.perf_counter()

                # Verify echo worked correctly
                if not np.array_equal(image, result):
                    print(f"   ❌ Image echo verification failed at sample {i}")
                    continue

                latency = end_time - start_time
                latencies.append(latency)

                # Calculate throughput (MB/s) - *2 for round-trip
                throughput = (image_size * 2) / latency / (1024 * 1024)
                throughputs.append(throughput)

                if (i + 1) % (self.samples // 4) == 0:
                    progress = (i + 1) / self.samples * 100
                    print(
                        f"   📈 Progress: {progress:.0f}% "
                        f"(latest: {latency * 1000:.1f}ms)"
                    )

            except Exception as e:
                print(f"   ⚠️  Sample {i} failed: {e}")
                continue

        return latencies, throughputs

    def _calculate_statistics(
        self,
        latencies: list[float],
        throughputs: list[float],
        width: int,
        height: int,
        channels: int,
        image_size: int
    ) -> Dict[str, Any]:
        """Calculate comprehensive statistics from collected samples.

        Args:
            latencies (list[float]): Collected latency measurements
            throughputs (list[float]): Collected throughput measurements
            width (int): Image width
            height (int): Image height
            channels (int): Number of color channels
            image_size (int): Image size in bytes

        Returns:
            Dict[str, Any]: Complete statistics dictionary
        """
        stats = {
            'resolution': (width, height, channels),
            'image_size_bytes': image_size,
            'image_size_formatted': self.format_size(image_size),
            'samples_collected': len(latencies),
            'latency_stats': {
                'mean_ms': statistics.mean(latencies) * 1000,
                'median_ms': statistics.median(latencies) * 1000,
                'min_ms': min(latencies) * 1000,
                'max_ms': max(latencies) * 1000,
                'std_ms': statistics.stdev(latencies) * 1000 if len(latencies) > 1 else 0,
                'p95_ms': np.percentile(latencies, 95) * 1000,
                'p99_ms': np.percentile(latencies, 99) * 1000,
            },
            'throughput_stats': {
                'mean_mbps': statistics.mean(throughputs),
                'median_mbps': statistics.median(throughputs),
                'min_mbps': min(throughputs),
                'max_mbps': max(throughputs),
                'std_mbps': statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
            }
        }
        return stats

    def _print_resolution_results(self, stats: Dict[str, Any]) -> None:
        """Print formatted results for a resolution test.

        Args:
            stats (Dict[str, Any]): Statistics dictionary to print
        """
        print("   ✅ Results:")
        print(
            f"      Latency: {stats['latency_stats']['mean_ms']:.1f}ms ± "
            f"{stats['latency_stats']['std_ms']:.1f}ms"
        )
        print(f"      Median:  {stats['latency_stats']['median_ms']:.1f}ms")
        print(
            f"      P95/P99: {stats['latency_stats']['p95_ms']:.1f}ms / "
            f"{stats['latency_stats']['p99_ms']:.1f}ms"
        )
        throughput_stats = stats['throughput_stats']['mean_mbps']
        print(
            f"      Throughput: {throughput_stats:.1f} MB/s ± "
            f"{stats['throughput_stats']['std_mbps']:.1f} MB/s"
        )

    def test_resolution(
        self, height: int, width: int, channels: int
    ) -> Optional[Dict[str, Any]]:
        """Test latency for a specific image resolution.

        Args:
            height (int): Image height
            width (int): Image width
            channels (int): Number of color channels

        Returns:
            Dict[str, Any]: Collected statistics for the resolution
        """
        print(f"\n📊 Testing resolution: {width}x{height}x{channels}")

        client = portal.Client(f'localhost:{self.port}', name='ImageClient')
        image_size = self.calculate_image_size(height, width, channels)
        print(f"   Image size: {self.format_size(image_size)}")

        # Perform warmup
        if not self._perform_warmup(client, height, width, channels):
            client.close()
            return None

        # Collect samples
        latencies, throughputs = self._collect_latency_samples(
            client, height, width, channels, image_size
        )
        client.close()

        # Check if we have valid measurements
        if not latencies:
            print(
                f"   ❌ No successful measurements for "
                f"{width}x{height}x{channels}"
            )
            return None

        # Calculate statistics
        stats = self._calculate_statistics(
            latencies, throughputs, width, height, channels, image_size
        )

        # Print results
        self._print_resolution_results(stats)

        return stats

    def _wait_for_server_ready(self) -> bool:
        """Wait for server to be ready and test connectivity.

        Returns:
            bool: True if server is ready, False otherwise
        """
        print("⏳ Waiting for server to be ready...")
        client = portal.Client(f'localhost:{self.port}', name='TestClient')
        res_msg = ""
        success = False
        N_ATTEMPTS = 10

        # Try to connect with retries
        attempt = 0
        while attempt < N_ATTEMPTS and not success:
            try:
                test_image = self.generate_random_image(32, 32, 3)
                info = client.get_image_info(test_image).result(timeout=5)
                res_msg = f"✅ Server ready! Test image info: {info}"
                success = True
            except Exception as e:
                attempt += 1
                if attempt < N_ATTEMPTS:
                    print(
                        f"   Attempt {attempt}/{N_ATTEMPTS} failed, retrying...")
                    time.sleep(1)
                else:
                    res_msg = (
                        "❌ Failed to connect to server after "
                        f"{N_ATTEMPTS} attempts: {e}"
                    )
                    success = False
            try:
                # Test server connectivity
                test_image = self.generate_random_image(32, 32, 3)
                info = client.get_image_info(test_image).result(timeout=5)
                res_msg = f"✅ Server ready! Test image info: {info}"
                success = True
            except Exception as e:
                res_msg = (
                    f"❌ Failed to connect to server after 10 attempts: {e}"
                )
                success = False
                if attempt < N_ATTEMPTS - 1:
                    print(
                        f"   Attempt {attempt + 1}/{N_ATTEMPTS} failed,"
                        " retrying..."
                    )
                    time.sleep(1)

        client.close()
        print(res_msg)
        return success

    def _run_resolution_tests(self) -> Dict[str, Any]:
        """Run latency tests for all configured resolutions.

        Returns:
            Dict[str, Any]: Test results for each resolution
        """
        results = {}
        total_start = time.time()

        for i, (width, height, channels) in enumerate(self.RESOLUTIONS):
            print(f"\n{'=' * 60}")
            print(f"Test {i + 1}/{len(self.RESOLUTIONS)}")

            result = self.test_resolution(height, width, channels)
            if result:
                results[f"{width}x{height}x{channels}"] = result

            # Progress update
            elapsed = time.time() - total_start
            if i < len(self.RESOLUTIONS) - 1:
                remaining = elapsed / (i + 1) * (len(self.RESOLUTIONS) - i - 1)
                print(
                    f"   ⏱️  Elapsed: {elapsed:.1f}s, "
                    f"Estimated remaining: {remaining:.1f}s"
                )

        total_elapsed = time.time() - total_start
        print(f"\n{'=' * 60}")
        print(f"🏁 All tests completed in {total_elapsed:.1f}s")

        return results

    def run_client_tests(self) -> Dict[str, Any]:
        """Run latency tests for all resolutions.

        Returns:
            Dict[str, Any]: Complete test results for all resolutions
        """
        print("🔬 Starting image latency tests")
        print(f"   Server: localhost:{self.port}")
        print(f"   Samples per resolution: {self.samples}")
        print(f"   Warmup rounds: {self.warmup}")

        # Wait for server to be ready
        if not self._wait_for_server_ready():
            return {}

        # Run tests for all resolutions
        results = self._run_resolution_tests()
        return results

    def print_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of all test results.

        Args:
            results (Dict[str, Any]): Complete test results for all resolutions
        """
        if not results:
            print("❌ No results to summarize")
            raise ValueError("No results to summarize")

        print(f"\n{'=' * 80}")
        print("📊 LATENCY TEST SUMMARY")
        print(f"{'=' * 80}")

        print(
            f"{'Resolution':<15} {'Size':<10} "
            f"{'Latency (ms)':<15} {'Throughput (MB/s)':<18} {'Samples':<8}"
        )
        print("-" * 80)

        for _, stats in results.items():
            res = (
                f"{stats['resolution'][0]}x"
                f"{stats['resolution'][1]}x"
                f"{stats['resolution'][2]}"
            )
            size = stats['image_size_formatted']
            latency = (
                f"{stats['latency_stats']['mean_ms']:.1f} ± "
                f"{stats['latency_stats']['std_ms']:.1f}"
            )
            throughput = (
                f"{stats['throughput_stats']['mean_mbps']:.1f} ± "
                f"{stats['throughput_stats']['std_mbps']:.1f}"
            )
            samples = stats['samples_collected']

            print(
                f"{res:<15} {size:<10} {latency:<15} "
                f"{throughput:<18} {samples:<8}"
            )

        # Find best/worst performers
        if len(results) > 1:
            print("\n📈 PERFORMANCE INSIGHTS:")

            valid_results = [(k, v) for k, v in results.items() if v]
            if valid_results:
                # Best latency
                best_latency = min(
                    valid_results,
                    key=lambda x: x[1]['latency_stats']['mean_ms'])
                print(
                    f"   🏆 Lowest latency: {best_latency[0]} ("
                    f"{best_latency[1]['latency_stats']['mean_ms']:.1f}ms)"
                )

            valid_results = [(k, v) for k, v in results.items() if v]
            if valid_results:
                # Best latency
                best_latency = min(
                    valid_results,
                    key=lambda x: x[1]['latency_stats']['mean_ms'])
                print(
                    f"   🏆 Lowest latency: {best_latency[0]} ("
                    f"{best_latency[1]['latency_stats']['mean_ms']:.1f}ms)"
                )

                # Best throughput
                best_throughput = max(
                    valid_results,
                    key=lambda x: x[1]['throughput_stats']['mean_mbps'])
                stat1 = best_throughput[0]
                stat2 = best_throughput[1]['throughput_stats']['mean_mbps']
                print(
                    f"   🚀 Highest throughput: {stat1} ({stat2:.1f} MB/s)"
                )

                # Latency vs size correlation
                sizes = [v['image_size_bytes'] for _, v in valid_results]
                latencies = [v['latency_stats']['mean_ms']
                             for _, v in valid_results]

                if len(sizes) > 2:
                    correlation = np.corrcoef(sizes, latencies)[0, 1]
                    print(f"   📊 Size-Latency correlation: {correlation:.3f}")

    def run_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run the complete test suite."""
        print("🌀 Portal Image Transfer Latency Test")
        print("=" * 50)

        # Start server process
        server_proc = portal.Process(
            self.server_function,
            name='ImageServer',
            start=True
        )

        try:
            # Run client tests
            self.results = self.run_client_tests()

            # Print summary
            self.print_summary(self.results)

        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
        finally:
            # Clean up
            print("\n🧹 Cleaning up...")
            server_proc.kill()
            if server_proc.running:
                server_proc.join(timeout=5)
            print("✅ Cleanup complete")

        return self.results


def main() -> int:
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Test image transfer latency using Portal')
    parser.add_argument(
        '--port',
        type=int,
        help='Port number for Portal server (auto-assigned if not specified)')
    parser.add_argument(
        '--samples',
        type=int,
        default=50,
        help='Number of samples per resolution (default: 50)')
    parser.add_argument(
        '--warmup',
        type=int,
        default=5,
        help='Number of warmup rounds (default: 5)')
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test with fewer samples and resolutions')

    args = parser.parse_args()

    # Adjust parameters for quick test
    if args.quick:
        print("🏃 Running quick test mode")
        args.samples = min(args.samples, 20)
        args.warmup = min(args.warmup, 3)

    # Create and run tester
    tester = ImageLatencyTester(
        port=args.port,
        samples=args.samples,
        warmup=args.warmup
    )

    # Limit resolutions for quick test
    if args.quick:
        # Only test first 4 resolutions
        tester.RESOLUTIONS = tester.RESOLUTIONS[:4]

    results = tester.run_tests()

    # Return success code based on results
    if results:
        print("\n✅ Test completed successfully!")
        return 0
    else:
        print("\n❌ Test failed or no results collected")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
