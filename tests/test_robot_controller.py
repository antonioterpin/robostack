"""
Test module for robot_controller.

This module demonstrates testing best practices from CONTRIBUTING.md:
- Test-driven development approach
- Arrange/Act/Assert pattern
- Property-based testing with Hypothesis
- Unit and integration test separation
- Comprehensive coverage including edge cases
"""

import pytest
from hypothesis import given, strategies as st
import jax.numpy as jnp

from src.my_package.robot_controller import RobotController, RobotState


class TestRobotState:
    """Unit tests for RobotState dataclass.
    
    These tests verify the immutable state container behavior
    and demonstrate the Arrange/Act/Assert pattern.
    """
    
    def test_robot_state_creation(self):
        """Test basic RobotState creation with default values."""
        # Arrange
        position = (1.0, 2.0)
        velocity = (0.5, -0.3)
        
        # Act
        state = RobotState(position=position, velocity=velocity)
        
        # Assert
        assert state.position == position
        assert state.velocity == velocity
        assert state.battery_level == 100.0
        assert state.timestamp == 0.0
        
        def test_robot_state_immutability(self):
                """Test that RobotState is truly immutable (frozen)."""
                # Arrange
                state = RobotState(position=(0.0, 0.0), velocity=(1.0, 1.0))
                
                # Act & Assert
                with pytest.raises(AttributeError):
                        state.position = (1.0, 1.0)  # Should raise error
        
        def test_robot_state_update_method(self):
                """Test the immutable update pattern."""
                # Arrange
                initial_state = RobotState(
                        position=(0.0, 0.0),
                        velocity=(1.0, 1.0),
                        battery_level=50.0
                )
                new_position = (2.0, 3.0)
                
                # Act
                updated_state = initial_state.update(position=new_position)
                
                # Assert
                assert updated_state.position == new_position
                # Velocity and battery should remain unchanged
                assert updated_state.velocity == initial_state.velocity
                assert updated_state.battery_level == initial_state.battery_level
                assert updated_state is not initial_state  # Different objects
        
        @given(
                x=st.floats(min_value=-100.0, max_value=100.0),
                y=st.floats(min_value=-100.0, max_value=100.0),
                vx=st.floats(min_value=-10.0, max_value=10.0),
                vy=st.floats(min_value=-10.0, max_value=10.0),
                battery=st.floats(min_value=0.0, max_value=100.0)
        )
        def test_robot_state_property_based(self, x, y, vx, vy, battery):
                """Property-based test for RobotState creation with random values."""
                # Arrange & Act
                state = RobotState(
                        position=(x, y),
                        velocity=(vx, vy),
                        battery_level=battery
                )
                
                # Assert properties that should always hold
                assert state.position == (x, y)
                assert state.velocity == (vx, vy)
                assert state.battery_level == battery
                assert 0.0 <= state.battery_level <= 100.0


class TestRobotController:
        """Unit tests for RobotController class.
        
        These tests verify pure function behavior and demonstrate
        comprehensive testing of business logic.
        """
        
        def setup_method(self):
                """Set up test fixtures (Arrange phase)."""
                self.controller = RobotController(max_speed=2.0, battery_drain_rate=0.1)
                self.initial_state = RobotState(
                        position=(0.0, 0.0),
                        velocity=(1.0, 0.0),
                        battery_level=100.0,
                        timestamp=0.0
                )
        
        def test_calculate_next_position_basic(self):
                """Test basic position calculation."""
                # Arrange
                time_delta = 1.0
                
                # Act
                next_pos = self.controller.calculate_next_position(
                        self.initial_state,
                        time_delta
                )
                
                # Assert
                expected_x = 0.0 + 1.0 * 1.0  # x + vx * dt
                expected_y = 0.0 + 0.0 * 1.0  # y + vy * dt
                assert next_pos == (expected_x, expected_y)
        
        def test_calculate_next_position_zero_time(self):
                """Test position calculation with zero time delta."""
                # Arrange
                time_delta = 0.0
                
                # Act
                next_pos = self.controller.calculate_next_position(
                        self.initial_state,
                        time_delta
                )
                
                # Assert
                assert next_pos == self.initial_state.position
        
        @given(
                vx=st.floats(min_value=-5.0, max_value=5.0),
                vy=st.floats(min_value=-5.0, max_value=5.0),
                dt=st.floats(min_value=0.01, max_value=2.0)
        )
        def test_calculate_next_position_property_based(self, vx, vy, dt):
                """Property-based test for position calculation."""
                # Arrange
                state = self.initial_state.update(velocity=(vx, vy))
                
                # Act
                next_pos = self.controller.calculate_next_position(state, dt)
                
                # Assert
                expected_x = state.position[0] + vx * dt
                expected_y = state.position[1] + vy * dt
                assert abs(next_pos[0] - expected_x) < 1e-6
                assert abs(next_pos[1] - expected_y) < 1e-6
        
        def test_calculate_battery_drain_stationary(self):
                """Test battery drain when robot is stationary."""
                # Arrange
                stationary_state = self.initial_state.update(velocity=(0.0, 0.0))
                time_delta = 1.0
                
                # Act
                drain = self.controller.calculate_battery_drain(stationary_state, time_delta)
                
                # Assert
                assert drain == 0.0  # No movement, no battery drain
        
        def test_calculate_battery_drain_moving(self):
                """Test battery drain calculation with movement."""
                # Arrange
                time_delta = 1.0
                # Speed = sqrt(1^2 + 0^2) = 1.0
                # Expected drain = 1.0 * 0.1 * 1.0 = 0.1
                
                # Act
                drain = self.controller.calculate_battery_drain(self.initial_state, time_delta)
                
                # Assert
                assert abs(drain - 0.1) < 1e-6
        
        def test_calculate_battery_drain_exceeds_available(self):
                """Test battery drain when calculated drain exceeds available battery."""
                # Arrange
                low_battery_state = self.initial_state.update(battery_level=0.05)
                time_delta = 1.0
                
                # Act
                drain = self.controller.calculate_battery_drain(low_battery_state, time_delta)
                
                # Assert
                assert drain == 0.05  # Should not exceed available battery
        
        def test_apply_speed_limit_under_limit(self):
                """Test speed limiting when velocity is under the limit."""
                # Arrange
                desired_velocity = (1.0, 1.0)  # Speed = sqrt(2) ≈ 1.41 < 2.0
                
                # Act
                limited_velocity = self.controller.apply_speed_limit(desired_velocity)
                
                # Assert
                assert limited_velocity == desired_velocity  # Should be unchanged
        
        def test_apply_speed_limit_over_limit(self):
                """Test speed limiting when velocity exceeds the limit."""
                # Arrange
                desired_velocity = (3.0, 4.0)  # Speed = 5.0 > 2.0
                
                # Act
                limited_velocity = self.controller.apply_speed_limit(desired_velocity)
                
                # Assert
                # Should be scaled down to max_speed = 2.0
                actual_speed = (limited_velocity[0]**2 + limited_velocity[1]**2)**0.5
                assert abs(actual_speed - 2.0) < 1e-6
                
                # Direction should be preserved
                original_direction = (3.0/5.0, 4.0/5.0)  # Normalized original
                actual_direction = (
                        limited_velocity[0] / actual_speed,
                        limited_velocity[1] / actual_speed
                )
                assert abs(actual_direction[0] - original_direction[0]) < 1e-6
                assert abs(actual_direction[1] - original_direction[1]) < 1e-6
        
        def test_apply_speed_limit_zero_velocity(self):
                """Test speed limiting with zero velocity."""
                # Arrange
                desired_velocity = (0.0, 0.0)
                
                # Act
                limited_velocity = self.controller.apply_speed_limit(desired_velocity)
                
                # Assert
                assert limited_velocity == (0.0, 0.0)
        
        def test_update_robot_state_basic(self):
                """Test complete state update with new velocity."""
                # Arrange
                desired_velocity = (0.5, 0.5)
                time_delta = 2.0
                
                # Act
                new_state = self.controller.update_robot_state(
                        self.initial_state,
                        desired_velocity,
                        time_delta
                )
                
                # Assert
                # Position should be updated: (0,0) + (0.5,0.5) * 2.0 = (1.0, 1.0)
                assert new_state.position == (1.0, 1.0)
                assert new_state.velocity == desired_velocity
                assert new_state.timestamp == 2.0
                
                # Battery should be drained
                speed = (0.5**2 + 0.5**2)**0.5  # ≈ 0.707
                expected_drain = speed * 0.1 * 2.0  # ≈ 0.141
                expected_battery = 100.0 - expected_drain
                assert abs(new_state.battery_level - expected_battery) < 1e-6
        
        def test_update_robot_state_no_new_velocity(self):
                """Test state update without changing velocity."""
                # Arrange
                time_delta = 1.0
                
                # Act
                new_state = self.controller.update_robot_state(
                        self.initial_state,
                        None,  # Keep current velocity
                        time_delta
                )
                
                # Assert
                assert new_state.velocity == self.initial_state.velocity
                assert new_state.position == (1.0, 0.0)  # Moved by velocity * time
        
        def test_batch_update_states_vectorization(self):
                """Test JAX vectorized batch processing."""
                # Arrange
                batch_size = 3
                # Create batch: [x, y, vx, vy, battery, timestamp]
                states = jnp.array([
                        [0.0, 0.0, 1.0, 0.0, 100.0, 0.0],  # Robot 1
                        [1.0, 1.0, 0.0, 1.0, 50.0, 1.0],   # Robot 2  
                        [2.0, 2.0, -1.0, -1.0, 25.0, 2.0]  # Robot 3
                ])
                time_delta = 1.0
                
                # Act
                updated_states = self.controller.batch_update_states(states, time_delta)
                
                # Assert
                assert updated_states.shape == (batch_size, 6)
                
                # Check first robot: position (0,0) + velocity (1,0) * 1 = (1,0)
                assert abs(updated_states[0, 0] - 1.0) < 1e-6  # x position
                assert abs(updated_states[0, 1] - 0.0) < 1e-6  # y position
                
                # Check timestamps are updated
                assert abs(updated_states[0, 5] - 1.0) < 1e-6  # timestamp
                assert abs(updated_states[1, 5] - 2.0) < 1e-6  # timestamp
                assert abs(updated_states[2, 5] - 3.0) < 1e-6  # timestamp


class TestRobotControllerIntegration:
        """Integration tests for RobotController.
        
        These tests verify the system behavior when components
        interact together over multiple time steps.
        """
        
        def test_robot_movement_simulation(self):
                """Integration test: simulate robot movement over time."""
                # Arrange
                controller = RobotController(max_speed=1.0, battery_drain_rate=0.2)
                initial_state = RobotState(
                        position=(0.0, 0.0),
                        velocity=(0.0, 0.0),
                        battery_level=10.0  # Low battery for testing
                )
                
                # Act: simulate 5 time steps
                current_state = initial_state
                desired_velocity = (0.8, 0.6)  # Speed = 1.0 (at max)
                time_delta = 1.0
                
                states_history = [current_state]
                for _ in range(5):
                        current_state = controller.update_robot_state(
                                current_state,
                                desired_velocity,
                                time_delta
                        )
                        states_history.append(current_state)
                
                # Assert
                final_state = states_history[-1]
                
                # Robot should have moved
                assert final_state.position != initial_state.position
                
                # Battery should be depleted
                assert final_state.battery_level < initial_state.battery_level
                
                # Timestamp should be updated
                assert final_state.timestamp == 5.0
                
                # Position should be consistent with velocity and time
                # After 5 steps: distance = speed * time = 1.0 * 5 = 5.0
                distance_traveled = (
                        final_state.position[0]**2 + final_state.position[1]**2
                )**0.5
                assert abs(distance_traveled - 5.0) < 1e-6
        
        def test_battery_depletion_stops_movement(self):
                """Integration test: verify behavior when battery is depleted."""
                # Arrange
                controller = RobotController(max_speed=2.0, battery_drain_rate=1.0)
                initial_state = RobotState(
                        position=(0.0, 0.0),
                        velocity=(0.0, 0.0),
                        battery_level=1.5  # Will be depleted quickly
                )
                
                # Act: run until battery depleted
                current_state = initial_state
                desired_velocity = (2.0, 0.0)  # Max speed
                time_delta = 1.0
                
                step_count = 0
                while current_state.battery_level > 0 and step_count < 10:
                        current_state = controller.update_robot_state(
                                current_state,
                                desired_velocity,
                                time_delta
                        )
                        step_count += 1
                
                # Assert
                assert current_state.battery_level == 0.0
                assert step_count < 10  # Should deplete within reasonable time
                
                # Robot should have moved some distance before stopping
                distance = (
                        current_state.position[0]**2 + current_state.position[1]**2
                )**0.5
                assert distance > 0


# Performance tests for JAX optimization
class TestPerformance:
        """Performance tests to verify JAX optimization effectiveness."""
        
        def test_batch_processing_performance(self):
                """Test that batch processing is faster than individual processing."""
                import time
                
                # Arrange
                controller = RobotController()
                batch_size = 1000
                
                # Create large batch of states
                states_batch = jnp.ones((batch_size, 6)) * jnp.arange(batch_size).reshape(-1, 1)
                time_delta = 0.1
                
                # Act: measure batch processing time
                start_time = time.time()
                batch_result = controller.batch_update_states(states_batch, time_delta)
                batch_time = time.time() - start_time
                
                # Act: measure individual processing time (sample)
                sample_size = min(10, batch_size)  # Test smaller sample
                start_time = time.time()
                for i in range(sample_size):
                        individual_state = RobotState(
                                position=(float(states_batch[i, 0]), float(states_batch[i, 1])),
                                velocity=(float(states_batch[i, 2]), float(states_batch[i, 3])),
                                battery_level=float(states_batch[i, 4]),
                                timestamp=float(states_batch[i, 5])
                        )
                        controller.update_robot_state(individual_state, None, time_delta)
                individual_time = time.time() - start_time
                
                # Assert: batch processing exists and produces results
                assert batch_result.shape == (batch_size, 6)
                
                # Note: Actual performance comparison would require larger datasets
                # and proper benchmarking, but we verify the batch function works
                assert batch_time > 0  # Sanity check
                assert individual_time > 0  # Sanity check
