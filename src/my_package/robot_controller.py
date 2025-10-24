"""
Robot controller module demonstrating RoboStack coding standards.

This module showcases the coding standards from CONTRIBUTING.md:
- 8-space indentation
- Single return statements at the end
- Immutable dataclasses with state separation
- Type hints everywhere
- Google-style docstrings
- JAX-optimized vectorized operations
"""

from dataclasses import dataclass, replace
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array


@dataclass(frozen=True)
class RobotState:
    """Immutable state container for robot data.
    
    This demonstrates the frozen dataclass pattern with state separation
    as required by the coding standards.
    
    Attributes:
        position: Robot position as (x, y) coordinates in meters
        velocity: Robot velocity as (vx, vy) in meters/second
        battery_level: Battery charge level from 0.0 to 100.0
        timestamp: Time since initialization in seconds
    """
    position: Tuple[float, float]
    velocity: Tuple[float, float] 
    battery_level: float = 100.0
    timestamp: float = 0.0
    
    def update(self, **kwargs) -> 'RobotState':
        """Create new state with updated fields.
        
        This method demonstrates the immutable update pattern using
        dataclasses.replace() as recommended in the coding standards.
        
        Args:
            **kwargs: Fields to update with new values
        
        Returns:
            RobotState: New state instance with updated values
        """
        result = replace(self, **kwargs)
        return result


class RobotController:
    """Robot controller with pure methods and external state.
    
    This class demonstrates:
    - State separation from business logic
    - Pure functions without side effects  
    - Single return statements at the end
    - JAX vectorization for performance
    - Comprehensive type hints
    """
    
    def __init__(self, max_speed: float = 2.0, battery_drain_rate: float = 0.1):
        """Initialize robot controller with configuration.
        
        Args:
            max_speed: Maximum allowed speed in meters/second
            battery_drain_rate: Battery drain per second of movement
        """
        self._max_speed = max_speed
        self._battery_drain_rate = battery_drain_rate
        
    def calculate_next_position(
        self,
        state: RobotState,
        time_delta: float
    ) -> Tuple[float, float]:
        """Calculate next position based on current state and time delta.
        
        This function demonstrates:
        - Pure function with no side effects
        - Single return statement at the end
        - Comprehensive type hints
        - JAX vectorization for performance
        
        Args:
            state: Current robot state with position and velocity
            time_delta: Time step in seconds for integration
        
        Returns:
            tuple[float, float]: Next position as (x, y) coordinates
        """
        # Convert to JAX arrays for vectorized computation
        pos_array = jnp.array(state.position)
        vel_array = jnp.array(state.velocity)
        
        # Vectorized position update using JAX
        next_pos_array = pos_array + vel_array * time_delta
        
        # Convert back to tuple for return
        result = (float(next_pos_array[0]), float(next_pos_array[1]))
        return result
        
    def calculate_battery_drain(
        self,
        state: RobotState,
        time_delta: float
    ) -> float:
        """Calculate battery drain based on movement.
        
        Args:
            state: Current robot state with velocity information
            time_delta: Time step in seconds
        
        Returns:
            float: Battery drain amount (0.0 to 100.0)
        """
        # Calculate speed magnitude using JAX for vectorization
        vel_array = jnp.array(state.velocity)
        speed = jnp.linalg.norm(vel_array)
        
        # Battery drain is proportional to speed and time
        drain = float(speed * self._battery_drain_rate * time_delta)
        result = min(drain, state.battery_level)
        return result
        
    def apply_speed_limit(
        self,
        desired_velocity: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Apply speed limit to desired velocity.
        
        This demonstrates JAX vectorization for mathematical operations
        while maintaining the pure function pattern.
        
        Args:
            desired_velocity: Desired velocity as (vx, vy)
        
        Returns:
            tuple[float, float]: Velocity limited to max_speed
        """
        vel_array = jnp.array(desired_velocity)
        speed = jnp.linalg.norm(vel_array)
        
        # Apply speed limit using JAX conditional operations
        limited_vel = jnp.where(
            speed > self._max_speed,
            vel_array * (self._max_speed / speed),
            vel_array
        )
        
        result = (float(limited_vel[0]), float(limited_vel[1]))
        return result
        
    def update_robot_state(
        self,
        state: RobotState,
        desired_velocity: Optional[Tuple[float, float]],
        time_delta: float
    ) -> RobotState:
        """Update robot state with new position and battery level.
        
        This is the main state update function that demonstrates:
        - Composition of pure functions
        - Immutable state updates
        - Single return statement
        - Proper error handling
        
        Args:
            state: Current robot state
            desired_velocity: Desired velocity or None to maintain current
            time_delta: Time step in seconds
        
        Returns:
            RobotState: New state with updated position, velocity, and battery
        """
        # Use current velocity if no new velocity specified
        if desired_velocity is None:
            current_velocity = state.velocity
        else:
            current_velocity = self.apply_speed_limit(desired_velocity)
        
        # Calculate next position using pure function
        next_position = self.calculate_next_position(
            state.update(velocity=current_velocity),
            time_delta
        )
        
        # Calculate battery drain
        battery_drain = self.calculate_battery_drain(
            state.update(velocity=current_velocity),
            time_delta
        )
        new_battery = max(0.0, state.battery_level - battery_drain)
        
        # Create new state with all updates (immutable pattern)
        result = state.update(
            position=next_position,
            velocity=current_velocity,
            battery_level=new_battery,
            timestamp=state.timestamp + time_delta
        )
        return result
        
    def batch_update_states(
        self,
        states: Array,
        time_delta: float
    ) -> Array:
        """Update multiple robot states in parallel using JAX vectorization.
        
        This demonstrates JAX vmap for batch processing as recommended
        in the performance guidelines.
        
        Args:
            states: Array of robot states with shape (batch_size, state_dim)
            time_delta: Time step in seconds
        
        Returns:
            Array: Updated states with same shape as input
        """
        # Extract positions and velocities for vectorized computation
        positions = states[:, :2]  # First 2 dims: x, y position
        velocities = states[:, 2:4]  # Next 2 dims: vx, vy velocity
        battery_levels = states[:, 4]  # 5th dim: battery level
        timestamps = states[:, 5]  # 6th dim: timestamp
        
        # Vectorized position updates
        next_positions = positions + velocities * time_delta
        
        # Vectorized battery drain calculation
        speeds = jnp.linalg.norm(velocities, axis=1)
        battery_drains = speeds * self._battery_drain_rate * time_delta
        new_battery_levels = jnp.maximum(0.0, battery_levels - battery_drains)
        new_timestamps = timestamps + time_delta
        
        # Combine all state components
        result = jnp.column_stack([
            next_positions,
            velocities,
            new_battery_levels,
            new_timestamps
        ])
        return result
