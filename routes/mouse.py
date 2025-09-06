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
        self.walls = set()  # Store wall information as (x1,y1,x2,y2) tuples
        self.visited = set()
        self.goal_cells = {(7,7), (7,8), (8,7), (8,8)}
        
        # Mouse state - track based on movement commands
        self.x, self.y = 0, 0
        self.direction = Direction.NORTH
        self.momentum = 0
        
        # Movement tracking
        self.last_instruction = None
        self.pending_movement = None  # Track expected movement
        
        # Strategy state
        self.current_strategy = "explore"
        self.runs_completed = 0
        self.last_run = -1
        self.instructions_this_request = 0
        
        # Add boundary walls
        self.add_boundary_walls()
        
        # Flood fill distances
        self.distances = [[float('inf') for _ in range(16)] for _ in range(16)]
        self.update_flood_fill()
        
        print(f"Controller initialized. Goal cells: {self.goal_cells}")

    def add_boundary_walls(self):
        """Add walls around the maze boundary"""
        for i in range(16):
            # Walls around the boundary
            self.walls.add((i, 15, i, 16))   # North boundary
            self.walls.add((i, -1, i, 0))    # South boundary  
            self.walls.add((-1, i, 0, i))    # West boundary
            self.walls.add((15, i, 16, i))   # East boundary

    def process_request(self, data: dict) -> dict:
        """Main controller logic"""
        
        # Extract request data
        sensor_data = data.get('sensor_data', [0, 0, 0, 0, 0])
        total_time_ms = data.get('total_time_ms', 0)
        goal_reached = data.get('goal_reached', False)
        best_time_ms = data.get('best_time_ms')
        run_time_ms = data.get('run_time_ms', 0)
        run = data.get('run', 0)
        momentum = data.get('momentum', 0)
        
        # Update momentum
        self.momentum = momentum
        
        print(f"Request - Position: ({self.x},{self.y}), Momentum: {momentum}, Goal reached: {goal_reached}, Run: {run}")
        print(f"Sensors: {sensor_data}")
        
        # Handle run changes
        if run > self.last_run:
            print(f"New run detected: {run}")
            self.last_run = run
            if run > 0:
                self.runs_completed += 1
            # Reset to start position
            self.x, self.y = 0, 0
            self.direction = Direction.NORTH
            print(f"Reset to start position")
        
        # Update walls based on sensor data
        self.update_walls_from_sensors(sensor_data)
        self.visited.add((self.x, self.y))
        
        # Strategy decision
        if self.runs_completed >= 2 and best_time_ms is not None:
            self.current_strategy = "speed_run"
        elif total_time_ms > 200000:  # Switch earlier to ensure completion
            self.current_strategy = "speed_run"
        
        print(f"Strategy: {self.current_strategy}")
        
        # Check if we should end
        if total_time_ms > 290000:
            print("Time limit approaching, ending")
            return {"instructions": [], "end": True}
        
        # Handle goal reached
        if goal_reached:
            print("Goal reached! Updating flood fill and path")
            self.update_flood_fill()
            self.find_optimal_path()
            # Let simulation handle return to start
            return {"instructions": [], "end": False}
        
        # Generate movement instructions
        instructions = self.get_movement_instructions()
        print(f"Generated instructions: {instructions}")
        
        return {"instructions": instructions, "end": False}

    def update_walls_from_sensors(self, sensor_data: List[int]):
        """Update wall map based on sensor data"""
        # Sensors: [-90°, -45°, 0°, +45°, +90°] relative to mouse direction
        sensor_directions = [
            (self.direction.value - 2) % 8,  # -90°
            (self.direction.value - 1) % 8,  # -45° 
            self.direction.value,             # 0° (forward)
            (self.direction.value + 1) % 8,  # +45°
            (self.direction.value + 2) % 8,  # +90°
        ]
        
        for i, has_wall in enumerate(sensor_data):
            if has_wall:
                sensor_dir = sensor_directions[i]
                self.add_wall_in_sensor_direction(sensor_dir)

    def add_wall_in_sensor_direction(self, sensor_dir: int):
        """Add wall in the direction of sensor detection"""
        # Convert sensor direction to movement delta
        direction_deltas = {
            0: (0, 1),   # North
            1: (1, 1),   # Northeast  
            2: (1, 0),   # East
            3: (1, -1),  # Southeast
            4: (0, -1),  # South
            5: (-1, -1), # Southwest
            6: (-1, 0),  # West
            7: (-1, 1),  # Northwest
        }
        
        if sensor_dir in direction_deltas:
            dx, dy = direction_deltas[sensor_dir]
            wall_x, wall_y = self.x + dx, self.y + dy
            
            # Add wall between current position and detected wall position
            wall = (min(self.x, wall_x), min(self.y, wall_y), 
                   max(self.x, wall_x), max(self.y, wall_y))
            self.walls.add(wall)
            print(f"Added wall: {wall}")

    def has_wall(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if there's a wall between two cells"""
        wall = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        return wall in self.walls

    def is_valid_move(self, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Check if move is valid (within bounds and no wall)"""
        if not (0 <= to_x < 16 and 0 <= to_y < 16):
            return False
        return not self.has_wall(from_x, from_y, to_x, to_y)

    def get_movement_instructions(self) -> List[str]:
        """Generate movement instructions based on current strategy"""
        
        # Update flood fill first
        self.update_flood_fill()
        
        print(f"Current distance to goal: {self.distances[self.x][self.y]}")
        
        # Find best next move
        next_move = self.find_best_next_move()
        
        if next_move is None:
            print("No valid moves found!")
            # Try to turn and explore
            if self.momentum == 0:
                return ["R"]  # Turn to look for new paths
            else:
                return ["F0"]  # Stop first
        
        target_x, target_y = next_move
        print(f"Moving toward: ({target_x}, {target_y})")
        
        return self.move_toward_cell(target_x, target_y)

    def find_best_next_move(self) -> Optional[Tuple[int, int]]:
        """Find the best next cell to move to"""
        current_dist = self.distances[self.x][self.y]
        candidates = []
        
        # Check all 4 cardinal directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # N, E, S, W
            target_x, target_y = self.x + dx, self.y + dy
            
            if self.is_valid_move(self.x, self.y, target_x, target_y):
                distance = self.distances[target_x][target_y]
                
                # Prioritize unexplored cells
                if (target_x, target_y) not in self.visited:
                    print(f"Found unexplored cell: ({target_x}, {target_y})")
                    return (target_x, target_y)
                
                # Otherwise, prefer cells closer to goal
                if distance < current_dist:
                    candidates.append((target_x, target_y, distance))
        
        # Sort by distance to goal
        if candidates:
            candidates.sort(key=lambda x: x[2])
            best_x, best_y, best_dist = candidates[0]
            print(f"Best candidate: ({best_x}, {best_y}) with distance {best_dist}")
            return (best_x, best_y)
        
        # If no better moves, try any valid move
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            target_x, target_y = self.x + dx, self.y + dy
            if self.is_valid_move(self.x, self.y, target_x, target_y):
                print(f"Fallback move: ({target_x}, {target_y})")
                return (target_x, target_y)
        
        return None

    def move_toward_cell(self, target_x: int, target_y: int) -> List[str]:
        """Generate instructions to move toward target cell"""
        
        dx = target_x - self.x
        dy = target_y - self.y
        
        # Determine required direction
        target_direction = None
        if dx == 0 and dy == 1:
            target_direction = Direction.NORTH
        elif dx == 1 and dy == 0:
            target_direction = Direction.EAST
        elif dx == 0 and dy == -1:
            target_direction = Direction.SOUTH
        elif dx == -1 and dy == 0:
            target_direction = Direction.WEST
        else:
            print(f"Invalid move: ({dx}, {dy})")
            return ["F0"] if self.momentum != 0 else ["R"]
        
        # Calculate turn needed
        current_dir = self.direction.value
        target_dir = target_direction.value
        turn_needed = (target_dir - current_dir) % 8
        
        print(f"Current dir: {current_dir}, Target dir: {target_dir}, Turn needed: {turn_needed}")
        
        # If we need to turn, do it first (must be at momentum 0)
        if turn_needed != 0:
            if self.momentum != 0:
                print("Need to stop before turning")
                return self.get_stop_instructions()
            
            # Turn toward target
            if turn_needed <= 4:
                # Turn clockwise (right)
                turns = turn_needed // 2  # Each R is 45 degrees
                if turns > 0:
                    print(f"Turning right {turns} times")
                    self.direction = target_direction  # Update our direction
                    return ["R"] * turns
            else:
                # Turn counter-clockwise (left)
                turns = (8 - turn_needed) // 2
                if turns > 0:
                    print(f"Turning left {turns} times")
                    self.direction = target_direction  # Update our direction
                    return ["L"] * turns
        
        # We're facing the right direction, now move forward
        print("Moving forward")
        
        # Choose acceleration based on strategy
        if self.current_strategy == "explore":
            max_speed = 2
        else:
            max_speed = 3
        
        # Update our position expectation
        self.x, self.y = target_x, target_y
        
        if self.momentum < max_speed:
            return ["F2"]  # Accelerate
        elif self.momentum > max_speed:
            return ["F0"]  # Decelerate
        else:
            return ["F1"]  # Maintain speed

    def get_stop_instructions(self) -> List[str]:
        """Get instructions to stop the mouse"""
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
        # Reset all distances
        for i in range(16):
            for j in range(16):
                self.distances[i][j] = float('inf')
        
        # Initialize goal cells
        queue = deque()
        for gx, gy in self.goal_cells:
            self.distances[gx][gy] = 0
            queue.append((gx, gy, 0))
        
        # Flood fill algorithm
        while queue:
            x, y, dist = queue.popleft()
            
            # Check all 4 cardinal neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < 16 and 0 <= ny < 16 and 
                    not self.has_wall(x, y, nx, ny) and 
                    self.distances[nx][ny] > dist + 1):
                    
                    self.distances[nx][ny] = dist + 1
                    queue.append((nx, ny, dist + 1))
        
        print(f"Flood fill complete. Start distance: {self.distances[0][0]}")

    def find_optimal_path(self):
        """Find optimal path from start to goal (for future speed runs)"""
        if self.distances[0][0] == float('inf'):
            print("No path to goal found!")
            return
        
        path = [(0, 0)]
        x, y = 0, 0
        
        while (x, y) not in self.goal_cells and len(path) < 100:
            best_next = None
            best_dist = float('inf')
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < 16 and 0 <= ny < 16 and 
                    not self.has_wall(x, y, nx, ny) and 
                    self.distances[nx][ny] < best_dist):
                    
                    best_dist = self.distances[nx][ny]
                    best_next = (nx, ny)
            
            if best_next is None:
                print("Path finding stuck!")
                break
            
            path.append(best_next)
            x, y = best_next
        
        print(f"Optimal path length: {len(path)}")


# Global controller instances
controllers = {}

@app.route('/micro-mouse', methods=['POST'])
def micro_mouse():
    """Main API endpoint"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        game_uuid = data.get('game_uuid', 'default')
        
        # Get or create controller
        if game_uuid not in controllers:
            controllers[game_uuid] = MicromouseController()
        
        controller = controllers[game_uuid]
        response = controller.process_request(data)
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error in micro_mouse: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "instructions": ["F0"], "end": False}), 200