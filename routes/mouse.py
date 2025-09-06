from flask import Flask, request, jsonify
import json
import math
from typing import List, Tuple, Dict, Optional, Set
from collections import deque
from enum import Enum

from routes import app

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
        self.walls = set()  # Store wall information as (x1,y1,x2,y2) tuples
        self.visited = set()
        self.goal_cells = {(7,7), (7,8), (8,7), (8,8)}
        
        # Mouse state - DO NOT UPDATE POSITION OURSELVES, rely on sensor feedback
        self.x, self.y = 0, 0  # Current position
        self.direction = Direction.NORTH
        self.momentum = 0
        
        # Strategy state
        self.exploration_complete = False
        self.optimal_path = []
        self.current_strategy = "explore"  # "explore" or "speed_run"
        self.runs_completed = 0
        self.last_run = -1
        
        # Add boundary walls
        self.add_boundary_walls()
        
        # Flood fill distances
        self.distances = [[float('inf') for _ in range(16)] for _ in range(16)]
        self.update_flood_fill()

    def add_boundary_walls(self):
        """Add walls around the maze boundary"""
        for i in range(16):
            # Top and bottom boundaries
            self.walls.add((i, 15, i, 16))  # Top
            self.walls.add((i, -1, i, 0))   # Bottom
            # Left and right boundaries  
            self.walls.add((-1, i, 0, i))   # Left
            self.walls.add((15, i, 16, i))  # Right

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
        
        # Detect new run (reset position when back at start)
        if run > self.last_run:
            self.last_run = run
            if run > 0:  # Not the first run
                self.runs_completed += 1
            # Reset position to start
            self.x, self.y = 0, 0
            self.direction = Direction.NORTH

        # Update walls based on sensor data (but don't move position)
        self.update_walls_from_sensors(sensor_data)
        self.visited.add((self.x, self.y))
        
        # Strategy decision
        if self.runs_completed >= 2 and best_time_ms is not None:
            self.current_strategy = "speed_run"
        elif total_time_ms > 250000:
            self.current_strategy = "speed_run"
        
        # Check if we should end
        if total_time_ms > 280000 or (best_time_ms and best_time_ms < 3000):
            return {"instructions": [], "end": True}
        
        # Handle goal reached
        if goal_reached:
            if self.current_strategy == "explore":
                self.update_flood_fill()
                self.find_optimal_path()
            return {"instructions": [], "end": False}  # Let simulation handle return to start
        
        # Generate movement instructions
        instructions = self.get_safe_instructions()
        
        return {"instructions": instructions, "end": False}

    def update_walls_from_sensors(self, sensor_data: List[int]):
        """Update wall map based on sensor data"""
        # Sensor angles relative to mouse direction: -90°, -45°, 0°, +45°, +90°
        sensor_angles = [-2, -1, 0, 1, 2]  # In 45-degree increments
        
        for i, has_wall in enumerate(sensor_data):
            if has_wall:
                # Calculate absolute direction of sensor
                sensor_dir_value = (self.direction.value + sensor_angles[i]) % 8
                
                # For wall detection, we only care about cardinal directions
                # Convert to nearest cardinal direction
                if sensor_dir_value in [7, 0, 1]:  # North-ish
                    self.add_wall(self.x, self.y, self.x, self.y + 1)
                elif sensor_dir_value in [1, 2, 3]:  # East-ish  
                    self.add_wall(self.x, self.y, self.x + 1, self.y)
                elif sensor_dir_value in [3, 4, 5]:  # South-ish
                    self.add_wall(self.x, self.y, self.x, self.y - 1)
                elif sensor_dir_value in [5, 6, 7]:  # West-ish
                    self.add_wall(self.x, self.y, self.x - 1, self.y)

    def add_wall(self, x1: int, y1: int, x2: int, y2: int):
        """Add wall between two cells"""
        wall = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        self.walls.add(wall)

    def has_wall(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if there's a wall between two adjacent cells"""
        wall = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        return wall in self.walls

    def is_valid_cell(self, x: int, y: int) -> bool:
        """Check if cell coordinates are valid"""
        return 0 <= x < 16 and 0 <= y < 16

    def can_move_to(self, target_x: int, target_y: int) -> bool:
        """Check if we can move to target cell (no wall blocking)"""
        if not self.is_valid_cell(target_x, target_y):
            return False
        return not self.has_wall(self.x, self.y, target_x, target_y)

    def get_safe_instructions(self) -> List[str]:
        """Generate safe movement instructions that won't crash into walls"""
        
        if self.current_strategy == "explore":
            return self.safe_explore_strategy()
        else:
            return self.safe_speed_strategy()

    def safe_explore_strategy(self) -> List[str]:
        """Safe exploration strategy"""
        # Update flood fill with current knowledge
        self.update_flood_fill()
        
        # Find best unexplored adjacent cell
        best_move = self.get_safe_exploration_move()
        if best_move:
            target_x, target_y = best_move
            return self.safe_move_to_adjacent_cell(target_x, target_y, fast=False)
        
        # If no unexplored cells, move toward goal
        goal_move = self.get_safe_goal_move()
        if goal_move:
            target_x, target_y = goal_move
            return self.safe_move_to_adjacent_cell(target_x, target_y, fast=False)
        
        # If stuck, try any valid move
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            target_x, target_y = self.x + dx, self.y + dy
            if self.can_move_to(target_x, target_y):
                return self.safe_move_to_adjacent_cell(target_x, target_y, fast=False)
        
        # Last resort - just try to turn or stop
        if self.momentum != 0:
            return ["F0"]  # Decelerate
        else:
            return ["R"]   # Turn to explore

    def safe_speed_strategy(self) -> List[str]:
        """Safe speed strategy using known optimal path"""
        self.update_flood_fill()
        
        # Try to follow optimal path if available
        if self.optimal_path:
            try:
                current_idx = self.optimal_path.index((self.x, self.y))
                if current_idx < len(self.optimal_path) - 1:
                    next_cell = self.optimal_path[current_idx + 1]
                    if self.can_move_to(*next_cell):
                        return self.safe_move_to_adjacent_cell(*next_cell, fast=True)
            except ValueError:
                pass
        
        # Fall back to safe goal movement
        goal_move = self.get_safe_goal_move()
        if goal_move:
            return self.safe_move_to_adjacent_cell(*goal_move, fast=True)
        
        # If no safe move toward goal, explore safely
        return self.safe_explore_strategy()

    def get_safe_exploration_move(self) -> Optional[Tuple[int, int]]:
        """Get next safe exploration move prioritizing unexplored cells"""
        candidates = []
        
        # Check all 4 cardinal directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            target_x, target_y = self.x + dx, self.y + dy
            
            if self.can_move_to(target_x, target_y):
                # Prioritize unexplored cells
                if (target_x, target_y) not in self.visited:
                    return (target_x, target_y)
                
                # Add to candidates with their flood fill distance
                distance = self.distances[target_x][target_y]
                candidates.append((target_x, target_y, distance))
        
        # If no unexplored cells, pick closest to goal
        if candidates:
            candidates.sort(key=lambda x: x[2])
            return (candidates[0][0], candidates[0][1])
        
        return None

    def get_safe_goal_move(self) -> Optional[Tuple[int, int]]:
        """Get next safe move toward goal"""
        current_dist = self.distances[self.x][self.y]
        best_move = None
        best_dist = current_dist
        
        # Check all 4 cardinal directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            target_x, target_y = self.x + dx, self.y + dy
            
            if self.can_move_to(target_x, target_y):
                distance = self.distances[target_x][target_y]
                if distance < best_dist:
                    best_dist = distance
                    best_move = (target_x, target_y)
        
        return best_move

    def safe_move_to_adjacent_cell(self, target_x: int, target_y: int, fast: bool = False) -> List[str]:
        """Generate safe movement to adjacent cell"""
        
        # Double-check the move is safe
        if not self.can_move_to(target_x, target_y):
            if self.momentum != 0:
                return ["F0"]  # Decelerate if we can't move
            else:
                return ["R"]   # Turn to look for other options
        
        dx = target_x - self.x
        dy = target_y - self.y
        
        # Determine required direction
        target_dir = None
        if dx == 0 and dy == 1:
            target_dir = Direction.NORTH
        elif dx == 1 and dy == 0:
            target_dir = Direction.EAST
        elif dx == 0 and dy == -1:
            target_dir = Direction.SOUTH
        elif dx == -1 and dy == 0:
            target_dir = Direction.WEST
        else:
            return ["F0"] if self.momentum != 0 else ["R"]
        
        instructions = []
        
        # Calculate turn needed
        turn_needed = (target_dir.value - self.direction.value) % 8
        
        # Handle turning - must be at momentum 0
        if turn_needed != 0:
            if self.momentum != 0:
                # Need to stop first
                return self.brake_instructions()
            
            # Turn toward target
            if turn_needed <= 4:
                # Turn right (clockwise)
                turns = turn_needed // 2
                for _ in range(turns):
                    instructions.append("R")
            else:
                # Turn left (counter-clockwise) 
                turns = (8 - turn_needed) // 2
                for _ in range(turns):
                    instructions.append("L")
            
            # Update our direction tracking
            self.direction = target_dir
            
            # Don't move this turn, just turn
            return instructions
        
        # We're facing the right direction, now move
        max_momentum = 3 if fast else 2
        
        if self.momentum < max_momentum:
            instructions.append("F2")  # Accelerate
        elif self.momentum > max_momentum:
            instructions.append("F0")  # Decelerate  
        else:
            instructions.append("F1")  # Maintain speed
        
        # Update our position tracking (the simulation will correct us if wrong)
        self.x, self.y = target_x, target_y
        
        return instructions

    def brake_instructions(self) -> List[str]:
        """Generate braking instructions"""
        if abs(self.momentum) >= 2:
            return ["BB"]
        elif self.momentum > 0:
            return ["F0"]
        elif self.momentum < 0:
            return ["V0"]
        else:
            return []

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
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (self.is_valid_cell(nx, ny) and 
                    not self.has_wall(x, y, nx, ny) and 
                    self.distances[nx][ny] > dist + 1):
                    
                    self.distances[nx][ny] = dist + 1
                    queue.append((nx, ny, dist + 1))

    def find_optimal_path(self):
        """Find optimal path from start to goal"""
        if self.distances[0][0] == float('inf'):
            return
        
        path = [(0, 0)]
        x, y = 0, 0
        
        while (x, y) not in self.goal_cells and len(path) < 50:  # Prevent infinite loops
            best_next = None
            best_dist = float('inf')
            
            # Check all 4 cardinal directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (self.is_valid_cell(nx, ny) and 
                    not self.has_wall(x, y, nx, ny) and 
                    self.distances[nx][ny] < best_dist):
                    
                    best_dist = self.distances[nx][ny]
                    best_next = (nx, ny)
            
            if best_next is None:
                break
            
            path.append(best_next)
            x, y = best_next
        
        self.optimal_path = path


# Global controller instances
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
        return jsonify({"error": str(e), "instructions": ["F0"], "end": False}), 200
