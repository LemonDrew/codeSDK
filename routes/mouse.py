from flask import Flask, request, jsonify
import json
import math
from typing import List, Tuple, Dict, Optional, Set
from collections import deque
import logging
from enum import Enum
from routes import app

logger = logging.getLogger(__name__)

class Direction(Enum):
    NORTH = 0
    NORTHEAST = 1
    EAST = 2
    SOUTHEAST = 3
    SOUTH = 4
    SOUTHWEST = 5
    WEST = 6
    NORTHWEST = 7

class MicromouseController:
    def __init__(self):
        # Maze representation (16x16 grid)
        self.maze = [[0 for _ in range(16)] for _ in range(16)]
        self.walls = {}  # Store wall information
        self.visited = set()
        self.goal_cells = {(7,7), (7,8), (8,7), (8,8)}
        
        # Mouse state
        self.x, self.y = 0, 0  # Current position
        self.direction = Direction.NORTH
        self.momentum = 0
        
        # Strategy state
        self.exploration_complete = False
        self.optimal_path = []
        self.current_strategy = "explore"  # "explore" or "speed_run"
        self.runs_completed = 0
        self.last_run = -1
        
        # Flood fill distances
        self.distances = [[float('inf') for _ in range(16)] for _ in range(16)]
        self.update_flood_fill()

    def process_request(self, data: dict) -> dict:
        """Main controller logic - processes API request and returns response"""
        
        # Extract request data
        game_uuid = data.get('game_uuid', '')
        sensor_data = data.get('sensor_data', [0, 0, 0, 0, 0])
        total_time_ms = data.get('total_time_ms', 0)
        goal_reached = data.get('goal_reached', False)
        best_time_ms = data.get('best_time_ms')
        run_time_ms = data.get('run_time_ms', 0)
        run = data.get('run', 0)
        momentum = data.get('momentum', 0)
        
        # Update momentum
        self.momentum = momentum
        
        # Detect new run
        if run > self.last_run:
            self.last_run = run
            if run > 0:  # Not the first run
                self.runs_completed += 1
                self.x, self.y = 0, 0  # Reset to start position
                self.direction = Direction.NORTH
                self.momentum = 0
        
        # Update position and walls based on sensor data
        self.update_position_and_walls(sensor_data)
        
        # Strategy decision
        if self.runs_completed >= 3 and best_time_ms is not None:
            # Switch to pure speed runs after 3 exploration runs
            self.current_strategy = "speed_run"
        elif total_time_ms > 250000:  # Save time for final runs
            self.current_strategy = "speed_run"
        
        # Check if we should end (time budget or excellent time achieved)
        if total_time_ms > 280000 or (best_time_ms and best_time_ms < 2500):
            return {"instructions": [], "end": True}
        
        # Handle goal reached
        if goal_reached:
            if self.current_strategy == "explore":
                self.update_flood_fill()
                self.find_optimal_path()
            # Return to start
            instructions = self.return_to_start()
            return {"instructions": instructions, "end": False}
        
        # Generate movement instructions
        if self.current_strategy == "explore":
            instructions = self.explore_strategy()
        else:
            instructions = self.speed_run_strategy()
        
        return {"instructions": instructions, "end": False}

    def update_position_and_walls(self, sensor_data: List[int]):
        """Update mouse position and wall map based on sensor data"""
        # Sensor positions: -90°, -45°, 0°, +45°, +90° relative to mouse
        sensor_angles = [-2, -1, 0, 1, 2]  # Relative to current direction
        
        for i, has_wall in enumerate(sensor_data):
            if has_wall:
                sensor_dir = (self.direction.value + sensor_angles[i]) % 8
                self.add_wall_in_direction(self.x, self.y, Direction(sensor_dir))
        
        self.visited.add((self.x, self.y))

    def add_wall_in_direction(self, x: int, y: int, direction: Direction):
        """Add wall information based on sensor detection"""
        dx, dy = self.get_direction_delta(direction)
        # Add wall between current cell and the cell in that direction
        nx, ny = x + dx, y + dy
        if 0 <= nx < 16 and 0 <= ny < 16:
            wall_key = (min(x, nx), min(y, ny), max(x, nx), max(y, ny))
            self.walls[wall_key] = True

    def has_wall(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if there's a wall between two adjacent cells"""
        wall_key = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        return wall_key in self.walls

    def get_direction_delta(self, direction: Direction) -> Tuple[int, int]:
        """Get x,y delta for a direction"""
        deltas = {
            Direction.NORTH: (0, 1), Direction.NORTHEAST: (1, 1),
            Direction.EAST: (1, 0), Direction.SOUTHEAST: (1, -1),
            Direction.SOUTH: (0, -1), Direction.SOUTHWEST: (-1, -1),
            Direction.WEST: (-1, 0), Direction.NORTHWEST: (-1, 1)
        }
        return deltas[direction]

    def update_flood_fill(self):
        """Update flood fill distances from goal"""
        # Reset distances
        for i in range(16):
            for j in range(16):
                self.distances[i][j] = float('inf')
        
        # Set goal distances to 0
        queue = deque()
        for gx, gy in self.goal_cells:
            self.distances[gx][gy] = 0
            queue.append((gx, gy, 0))
        
        # Flood fill
        while queue:
            x, y, dist = queue.popleft()
            
            # Check all 4 cardinal directions
            for direction in [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]:
                dx, dy = self.get_direction_delta(direction)
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < 16 and 0 <= ny < 16 and 
                    not self.has_wall(x, y, nx, ny) and 
                    self.distances[nx][ny] > dist + 1):
                    
                    self.distances[nx][ny] = dist + 1
                    queue.append((nx, ny, dist + 1))

    def find_optimal_path(self):
        """Find optimal path from start to goal using current knowledge"""
        if self.distances[0][0] == float('inf'):
            return
        
        path = [(0, 0)]
        x, y = 0, 0
        
        while (x, y) not in self.goal_cells:
            best_next = None
            best_dist = float('inf')
            
            # Check all 4 cardinal directions
            for direction in [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]:
                dx, dy = self.get_direction_delta(direction)
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < 16 and 0 <= ny < 16 and 
                    not self.has_wall(x, y, nx, ny) and 
                    self.distances[nx][ny] < best_dist):
                    
                    best_dist = self.distances[nx][ny]
                    best_next = (nx, ny)
            
            if best_next is None:
                break
            
            path.append(best_next)
            x, y = best_next
        
        self.optimal_path = path

    def explore_strategy(self) -> List[str]:
        """Exploration strategy using flood fill"""
        # Find next cell with minimum distance that we haven't fully explored
        best_move = self.get_flood_fill_move()
        if best_move:
            return self.move_to_adjacent_cell(*best_move, fast=False)
        
        # If no good exploration move, head towards goal
        return self.move_towards_goal()

    def get_flood_fill_move(self) -> Optional[Tuple[int, int]]:
        """Get next move based on flood fill algorithm"""
        current_dist = self.distances[self.x][self.y]
        best_cells = []
        
        # Check all 4 cardinal directions
        for direction in [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]:
            dx, dy = self.get_direction_delta(direction)
            nx, ny = self.x + dx, self.y + dy
            
            if (0 <= nx < 16 and 0 <= ny < 16 and 
                not self.has_wall(self.x, self.y, nx, ny)):
                
                cell_dist = self.distances[nx][ny]
                # Prefer unexplored cells or cells closer to goal
                if (nx, ny) not in self.visited:
                    return (nx, ny)
                elif cell_dist < current_dist:
                    best_cells.append((nx, ny, cell_dist))
        
        if best_cells:
            best_cells.sort(key=lambda x: x[2])
            return (best_cells[0][0], best_cells[0][1])
        
        return None

    def speed_run_strategy(self) -> List[str]:
        """High-speed strategy using optimal path"""
        if not self.optimal_path:
            return self.move_towards_goal()
        
        # Find current position in optimal path
        try:
            current_idx = self.optimal_path.index((self.x, self.y))
            if current_idx < len(self.optimal_path) - 1:
                next_cell = self.optimal_path[current_idx + 1]
                return self.move_to_adjacent_cell(*next_cell, fast=True)
        except ValueError:
            pass
        
        return self.move_towards_goal()

    def move_towards_goal(self) -> List[str]:
        """Move towards goal using flood fill distances"""
        best_move = self.get_flood_fill_move()
        if best_move:
            return self.move_to_adjacent_cell(*best_move, fast=self.current_strategy == "speed_run")
        return ["F1"]  # Default: move forward slowly

    def move_to_adjacent_cell(self, target_x: int, target_y: int, fast: bool = False) -> List[str]:
        """Generate movement commands to reach adjacent cell"""
        dx = target_x - self.x
        dy = target_y - self.y
        
        # Calculate required direction (only cardinal directions for movement)
        if dx == 0 and dy == 1:
            target_dir = Direction.NORTH
        elif dx == 1 and dy == 0:
            target_dir = Direction.EAST
        elif dx == 0 and dy == -1:
            target_dir = Direction.SOUTH
        elif dx == -1 and dy == 0:
            target_dir = Direction.WEST
        else:
            return ["F1"]  # Invalid move, default action
        
        instructions = []
        
        # Calculate turn needed (in 45-degree increments)
        current_cardinal = self.get_nearest_cardinal(self.direction)
        turn_needed = (target_dir.value - current_cardinal.value) % 8
        
        # Handle turning - must be at momentum 0
        if turn_needed != 0:
            if self.momentum != 0:
                # Need to stop first
                instructions.extend(self.brake_to_stop())
            
            # Optimize turning direction
            if turn_needed <= 4:
                # Turn right (clockwise)
                for _ in range(turn_needed // 2):  # Each R is 45 degrees
                    instructions.append("R")
            else:
                # Turn left (counter-clockwise) - shorter path
                for _ in range((8 - turn_needed) // 2):
                    instructions.append("L")
            
            self.direction = target_dir
        
        # Handle forward movement with momentum optimization
        max_momentum = 4 if fast else 2
        
        if self.momentum < max_momentum:
            # Accelerate
            instructions.append("F2")
        elif self.momentum > max_momentum:
            # Decelerate
            instructions.append("F0")
        else:
            # Maintain speed
            instructions.append("F1")
        
        # Update position (this is a prediction, actual position updated by sensor data)
        self.x, self.y = target_x, target_y
        
        return instructions

    def get_nearest_cardinal(self, direction: Direction) -> Direction:
        """Get nearest cardinal direction"""
        cardinal_dirs = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
        angles = [0, 2, 4, 6]  # NORTH, EAST, SOUTH, WEST in 45-degree units
        
        dir_angle = direction.value
        min_diff = float('inf')
        nearest = Direction.NORTH
        
        for i, angle in enumerate(angles):
            diff = min(abs(dir_angle - angle), 8 - abs(dir_angle - angle))
            if diff < min_diff:
                min_diff = diff
                nearest = cardinal_dirs[i]
        
        return nearest

    def brake_to_stop(self) -> List[str]:
        """Generate commands to stop the mouse"""
        instructions = []
        temp_momentum = self.momentum
        
        while temp_momentum != 0:
            if abs(temp_momentum) >= 2:
                instructions.append("BB")
                if temp_momentum > 0:
                    temp_momentum = max(0, temp_momentum - 2)
                else:
                    temp_momentum = min(0, temp_momentum + 2)
            else:
                if temp_momentum > 0:
                    instructions.append("F0")
                    temp_momentum -= 1
                else:
                    instructions.append("V0")
                    temp_momentum += 1
        
        return instructions

    def return_to_start(self) -> List[str]:
        """Generate commands to return to start position"""
        # If already at start with no momentum, stay put
        if (self.x, self.y) == (0, 0) and self.momentum == 0:
            return []
        
        # If in goal area, just stop for now (simple strategy)
        if (self.x, self.y) in self.goal_cells:
            return self.brake_to_stop()
        
        # Move towards start (simplified - could be optimized)
        if self.x > 0:
            return self.move_to_adjacent_cell(self.x - 1, self.y)
        elif self.y > 0:
            return self.move_to_adjacent_cell(self.x, self.y - 1)
        
        return []


# Global controller instance (per game session)
controllers = {}

@app.route('/micro-mouse', methods=['POST'])
def micro_mouse():
    """Main API endpoint for micromouse controller"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        game_uuid = data.get('game_uuid', 'default')
        
        # Get or create controller for this game
        if game_uuid not in controllers:
            controllers[game_uuid] = MicromouseController()
        
        controller = controllers[game_uuid]
        
        # Process the request
        response = controller.process_request(data)
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
