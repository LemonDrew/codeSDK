from flask import Flask, request, jsonify
import json
import math
from typing import List, Tuple, Dict, Optional, Set
from collections import deque
from enum import Enum

app = Flask(__name__)

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
        
        # Mouse state - CORRECTED: starts at (0,15) top-left, goal is center
        self.x, self.y = 0, 15  # Start at top-left corner
        self.direction = Direction.NORTH  # Facing up initially
        self.momentum = 0
        
        # Movement tracking
        self.last_instruction = None
        self.pending_movement = None
        
        # Strategy state
        self.current_strategy = "explore"
        self.runs_completed = 0
        self.last_run = -1
        self.move_count = 0
        
        # Flood fill distances
        self.distances = [[float('inf') for _ in range(16)] for _ in range(16)]
        self.update_flood_fill()
        
        print(f"Controller initialized. Start: ({self.x},{self.y}), Goal cells: {self.goal_cells}")

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
        
        print(f"\n=== REQUEST {self.move_count} ===")
        print(f"Position: ({self.x},{self.y}), Momentum: {momentum}, Goal reached: {goal_reached}, Run: {run}")
        print(f"Sensors: {sensor_data} (L45, L, Forward, R, R45)")
        self.move_count += 1
        
        # Handle run changes
        if run > self.last_run:
            print(f"New run detected: {run}")
            self.last_run = run
            if run > 0:
                self.runs_completed += 1
            # Reset to start position (top-left)
            self.x, self.y = 0, 15
            self.direction = Direction.NORTH
            print(f"Reset to start position (0,15)")
        
        # Update walls based on sensor data
        self.update_walls_from_sensors(sensor_data)
        self.visited.add((self.x, self.y))
        
        # Strategy decision
        if self.runs_completed >= 2 and best_time_ms is not None:
            self.current_strategy = "speed_run"
        elif total_time_ms > 200000:
            self.current_strategy = "speed_run"
        
        print(f"Strategy: {self.current_strategy}")
        
        # Check if we should end
        if total_time_ms > 290000:
            print("Time limit approaching, ending")
            return {"instructions": [], "end": True}
        
        # Handle goal reached
        if goal_reached:
            print("Goal reached! Updating flood fill")
            self.update_flood_fill()
            return {"instructions": [], "end": False}
        
        # Generate movement instructions
        instructions = self.get_movement_instructions()
        print(f"Generated instructions: {instructions}")
        
        return {"instructions": instructions, "end": False}

    def update_walls_from_sensors(self, sensor_data: List[int]):
        """Update wall map based on sensor data"""
        # Sensors: [-90°, -45°, 0°, +45°, +90°] relative to mouse direction
        # Let's be more careful about wall placement
        
        directions = ['L90', 'L45', 'Forward', 'R45', 'R90']
        
        for i, has_wall in enumerate(sensor_data):
            if has_wall:
                # Calculate the direction the sensor is pointing
                sensor_angle = (self.direction.value + [-2, -1, 0, 1, 2][i]) % 8
                
                # Get the adjacent cell in that direction
                dx, dy = self.get_direction_delta(sensor_angle)
                adj_x, adj_y = self.x + dx, self.y + dy
                
                # Only add wall if the adjacent cell would be in bounds
                # Wall is between current cell and adjacent cell
                if 0 <= adj_x < 16 and 0 <= adj_y < 16:
                    wall = (min(self.x, adj_x), min(self.y, adj_y), 
                           max(self.x, adj_x), max(self.y, adj_y))
                    self.walls.add(wall)
                    print(f"Added internal wall {directions[i]}: {wall}")
                else:
                    # This is a boundary wall - expected
                    print(f"Detected boundary wall {directions[i]} at ({adj_x},{adj_y})")

    def get_direction_delta(self, direction_value: int) -> Tuple[int, int]:
        """Get x,y delta for a direction value (0-7)"""
        deltas = [
            (0, 1),   # 0: North
            (1, 1),   # 1: Northeast  
            (1, 0),   # 2: East
            (1, -1),  # 3: Southeast
            (0, -1),  # 4: South
            (-1, -1), # 5: Southwest
            (-1, 0),  # 6: West
            (-1, 1),  # 7: Northwest
        ]
        return deltas[direction_value % 8]

    def has_wall(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        """Check if there's a wall between two cells"""
        wall = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        return wall in self.walls

    def is_valid_move(self, from_x: int, from_y: int, to_x: int, to_y: int) -> bool:
        """Check if move is valid (within bounds and no wall)"""
        # Check bounds
        if not (0 <= to_x < 16 and 0 <= to_y < 16):
            return False
        
        # Check for wall
        if self.has_wall(from_x, from_y, to_x, to_y):
            return False
        
        return True

    def get_movement_instructions(self) -> List[str]:
        """Generate movement instructions based on current strategy"""
        
        # Update flood fill first
        self.update_flood_fill()
        
        current_dist = self.distances[self.x][self.y]
        print(f"Current distance to goal: {current_dist}")
        
        # Check all possible moves
        possible_moves = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:  # N, E, S, W
            target_x, target_y = self.x + dx, self.y + dy
            if self.is_valid_move(self.x, self.y, target_x, target_y):
                distance = self.distances[target_x][target_y]
                is_unexplored = (target_x, target_y) not in self.visited
                possible_moves.append((target_x, target_y, distance, is_unexplored))
                print(f"Possible move to ({target_x},{target_y}): distance={distance}, unexplored={is_unexplored}")
        
        if not possible_moves:
            print("No valid moves found!")
            # Try to turn and see if that helps
            if self.momentum == 0:
                return ["R"]  # Turn to potentially see new areas
            else:
                return ["F0"]  # Stop first
        
        # Sort moves: unexplored first, then by distance to goal
        possible_moves.sort(key=lambda x: (not x[3], x[2]))  # unexplored first, then shortest distance
        
        target_x, target_y, _, _ = possible_moves[0]
        print(f"Selected move to: ({target_x}, {target_y})")
        
        return self.move_toward_cell(target_x, target_y)

    def move_toward_cell(self, target_x: int, target_y: int) -> List[str]:
        """Generate instructions to move toward target cell"""
        
        dx = target_x - self.x
        dy = target_y - self.y
        
        # Determine required direction
        target_direction = None
        direction_name = ""
        if dx == 0 and dy == 1:
            target_direction = Direction.NORTH
            direction_name = "North"
        elif dx == 1 and dy == 0:
            target_direction = Direction.EAST
            direction_name = "East"
        elif dx == 0 and dy == -1:
            target_direction = Direction.SOUTH
            direction_name = "South"
        elif dx == -1 and dy == 0:
            target_direction = Direction.WEST
            direction_name = "West"
        else:
            print(f"Invalid move delta: ({dx}, {dy})")
            return ["F0"] if self.momentum != 0 else ["R"]
        
        print(f"Need to move {direction_name}")
        
        # Calculate turn needed
        current_dir = self.direction.value
        target_dir = target_direction.value
        turn_needed = (target_dir - current_dir) % 8
        
        print(f"Current facing: {self.direction.name} ({current_dir})")
        print(f"Target facing: {target_direction.name} ({target_dir})")
        print(f"Turn needed: {turn_needed} (45° increments)")
        
        # If we need to turn, do it first (must be at momentum 0)
        if turn_needed != 0:
            if self.momentum != 0:
                print("Need to stop before turning")
                return ["F0"]  # Decelerate first
            
            # Turn toward target (each turn is 45°)
            if turn_needed <= 4:
                # Turn clockwise (right)
                turns = turn_needed // 2  # Each R is 45 degrees, so divide by 2 if we need 90°+ 
                if turn_needed % 2 == 1:  # Odd number means we need one 45° turn
                    print(f"Turning right 45° (1 R command)")
                    self.direction = Direction((self.direction.value + 1) % 8)
                    return ["R"]
                else:
                    print(f"Turning right {turn_needed * 45}° ({turns} R commands)")
                    self.direction = target_direction
                    return ["R"] * turns
            else:
                # Turn counter-clockwise (left) - shorter path
                left_turns = (8 - turn_needed)
                turns = left_turns // 2
                if left_turns % 2 == 1:
                    print(f"Turning left 45° (1 L command)")
                    self.direction = Direction((self.direction.value - 1) % 8)
                    return ["L"]
                else:
                    print(f"Turning left {left_turns * 45}° ({turns} L commands)")
                    self.direction = target_direction
                    return ["L"] * turns
        
        # We're facing the right direction, now move forward
        print(f"Moving forward from ({self.x},{self.y}) to ({target_x},{target_y})")
        
        # Update our position expectation
        self.x, self.y = target_x, target_y
        
        # Choose speed based on strategy
        if self.current_strategy == "explore":
            if self.momentum < 2:
                return ["F2"]  # Accelerate
            else:
                return ["F1"]  # Maintain moderate speed
        else:
            if self.momentum < 3:
                return ["F2"]  # Accelerate for speed runs
            else:
                return ["F1"]  # Maintain higher speed

    def update_flood_fill(self):
        """Update flood fill distances from goal"""
        # Reset all distances
        for i in range(16):
            for j in range(16):
                self.distances[i][j] = float('inf')
        
        # Initialize goal cells with distance 0
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
        
        start_distance = self.distances[0][15]  # Distance from actual start position
        print(f"Flood fill complete. Start distance: {start_distance}")
        
        # Debug: print some distances
        print(f"Distance to goal from current position ({self.x},{self.y}): {self.distances[self.x][self.y]}")


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