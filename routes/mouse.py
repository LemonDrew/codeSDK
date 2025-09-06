from flask import request, jsonify
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
        
        # Mouse state - starts at bottom-left (0,0), facing North
        self.x, self.y = 0, 0  # Actually start at bottom-left!
        self.direction = Direction.NORTH  # Facing North (toward positive Y)
        self.momentum = 0
        
        # Strategy state
        self.current_strategy = "explore"
        self.runs_completed = 0
        self.last_run = -1
        self.move_count = 0
        self.stuck_counter = 0
        
        # Add boundary walls
        self.add_boundary_walls()
        
        # Flood fill distances
        self.distances = [[float('inf') for _ in range(16)] for _ in range(16)]
        self.update_flood_fill()
        
        print(f"Controller initialized. Start: ({self.x},{self.y}), Goal cells: {self.goal_cells}")

    def add_boundary_walls(self):
        """Add walls around the maze boundary"""
        for i in range(16):
            # Boundary walls - these prevent moving outside the maze
            self.walls.add((i, 15, i, 16))   # Top boundary (y=16)
            self.walls.add((i, -1, i, 0))    # Bottom boundary (y=-1)
            self.walls.add((-1, i, 0, i))    # Left boundary (x=-1)
            self.walls.add((15, i, 16, i))   # Right boundary (x=16)

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
        print(f"Position: ({self.x},{self.y}), Momentum: {momentum}, Direction: {self.direction.name}")
        print(f"Goal reached: {goal_reached}, Run: {run}")
        print(f"Sensors: {sensor_data} (L90, L45, Forward, R45, R90)")
        self.move_count += 1
        
        # Handle run changes (reset to start)
        if run > self.last_run:
            print(f"New run detected: {run}")
            self.last_run = run
            if run > 0:
                self.runs_completed += 1
            # Reset to start position
            self.x, self.y = 0, 0
            self.direction = Direction.NORTH
            self.stuck_counter = 0
            print(f"Reset to start position (0,0)")
        
        # Update walls based on sensor data
        self.update_walls_from_sensors(sensor_data)
        self.visited.add((self.x, self.y))
        
        # Check if we should end
        if total_time_ms > 290000 or self.stuck_counter > 20:
            print("Time limit or stuck too long, ending")
            return {"instructions": [], "end": True}
        
        # Handle goal reached
        if goal_reached:
            print("Goal reached! Updating flood fill")
            self.update_flood_fill()
            self.stuck_counter = 0
            return {"instructions": [], "end": False}
        
        # Generate movement instructions
        instructions = self.get_movement_instructions()
        print(f"Generated instructions: {instructions}")
        
        return {"instructions": instructions, "end": False}

    def update_walls_from_sensors(self, sensor_data: List[int]):
        """Update wall map based on sensor data - ONLY add walls where sensors detect them"""
        
        # Sensor directions relative to current facing direction
        # [-90°, -45°, 0°, +45°, +90°]
        relative_angles = [-2, -1, 0, 1, 2]
        sensor_names = ['L90', 'L45', 'Forward', 'R45', 'R90']
        
        walls_added = False
        for i, has_wall in enumerate(sensor_data):
            if has_wall:
                # Calculate absolute direction of this sensor
                sensor_direction = (self.direction.value + relative_angles[i]) % 8
                
                # Get the cell this sensor is looking toward
                dx, dy = self.get_direction_delta(sensor_direction)
                adjacent_x = self.x + dx
                adjacent_y = self.y + dy
                
                # Only add walls for cells that are within the maze bounds
                if 0 <= adjacent_x < 16 and 0 <= adjacent_y < 16:
                    # There's a wall between current cell and the adjacent cell
                    wall = (min(self.x, adjacent_x), min(self.y, adjacent_y), 
                           max(self.x, adjacent_x), max(self.y, adjacent_y))
                    if wall not in self.walls:
                        self.walls.add(wall)
                        print(f"Added wall {sensor_names[i]}: between ({self.x},{self.y}) and ({adjacent_x},{adjacent_y})")
                        walls_added = True
                else:
                    print(f"Sensor {sensor_names[i]} detected boundary wall at ({adjacent_x},{adjacent_y})")
        
        if not walls_added:
            print("No new walls detected this turn")

    def get_direction_delta(self, direction_value: int) -> Tuple[int, int]:
        """Get x,y delta for a direction value (0-7)"""
        deltas = [
            (0, 1),   # 0: North (+Y)
            (1, 1),   # 1: Northeast  
            (1, 0),   # 2: East (+X)
            (1, -1),  # 3: Southeast
            (0, -1),  # 4: South (-Y)
            (-1, -1), # 5: Southwest
            (-1, 0),  # 6: West (-X)
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
        
        # Find all possible moves
        possible_moves = []
        cardinal_directions = [
            (0, 1, "North"), 
            (1, 0, "East"), 
            (0, -1, "South"), 
            (-1, 0, "West")
        ]
        
        for dx, dy, name in cardinal_directions:
            target_x, target_y = self.x + dx, self.y + dy
            
            if self.is_valid_move(self.x, self.y, target_x, target_y):
                distance = self.distances[target_x][target_y]
                is_unexplored = (target_x, target_y) not in self.visited
                possible_moves.append((target_x, target_y, distance, is_unexplored, name))
                print(f"Can move {name} to ({target_x},{target_y}): distance={distance}, unexplored={is_unexplored}")
        
        if not possible_moves:
            print("No valid moves available!")
            self.stuck_counter += 1
            
            # Try to turn to see new areas
            if self.momentum == 0:
                return ["R"]  # Turn right to explore
            else:
                return ["F0"]  # Stop first
        
        # Reset stuck counter since we found moves
        self.stuck_counter = 0
        
        # Prioritize unexplored cells, then shortest distance to goal
        possible_moves.sort(key=lambda x: (not x[3], x[2]))
        
        target_x, target_y, target_dist, is_unexplored, direction_name = possible_moves[0]
        print(f"Selected: Move {direction_name} to ({target_x},{target_y}) - distance={target_dist}, unexplored={is_unexplored}")
        
        return self.move_toward_cell(target_x, target_y)

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
            print(f"Invalid move delta: ({dx}, {dy})")
            return ["F0"] if self.momentum != 0 else ["R"]
        
        # Calculate turn needed (in 45-degree increments)
        current_dir = self.direction.value
        target_dir = target_direction.value
        turn_needed = (target_dir - current_dir) % 8
        
        print(f"Current: {self.direction.name}({current_dir}), Target: {target_direction.name}({target_dir}), Turn: {turn_needed}")
        
        # If we need to turn, do it first (must be at momentum 0)
        if turn_needed != 0:
            if self.momentum != 0:
                print("Must stop before turning")
                return ["F0"]
            
            # Turn toward target
            if turn_needed == 2:  # 90 degrees clockwise
                print("Turning right 90 degrees")
                self.direction = target_direction
                return ["R", "R"]
            elif turn_needed == 6:  # 90 degrees counter-clockwise (270 clockwise)
                print("Turning left 90 degrees") 
                self.direction = target_direction
                return ["L", "L"]
            elif turn_needed == 4:  # 180 degrees
                print("Turning around 180 degrees")
                self.direction = target_direction
                return ["R", "R", "R", "R"]
            else:
                # Handle other angles
                if turn_needed <= 4:
                    turns = turn_needed // 2
                    remaining = turn_needed % 2
                    commands = ["R"] * turns
                    if remaining:
                        commands.append("R")
                    self.direction = target_direction
                    return commands
                else:
                    turns = (8 - turn_needed) // 2
                    remaining = (8 - turn_needed) % 2
                    commands = ["L"] * turns
                    if remaining:
                        commands.append("L")
                    self.direction = target_direction
                    return commands
        
        # We're facing the right direction, now move forward
        print(f"Moving forward from ({self.x},{self.y}) to ({target_x},{target_y})")
        
        # Update our position expectation
        self.x, self.y = target_x, target_y
        
        # Move forward with appropriate speed
        if self.momentum < 2:
            return ["F2"]  # Accelerate
        else:
            return ["F1"]  # Maintain speed

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
        
        start_distance = self.distances[0][0]  # Distance from start position
        print(f"Flood fill complete. Start distance: {start_distance}")


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

if __name__ == '__main__':
    print("Micromouse Controller loaded!")
    print("Fixed coordinate system - start at (0,0) bottom-left!")