import math
from flask import Flask, request, jsonify
from routes import app

# ---------------------------------------------------------------------------- #
#                                  Mouse Class                                 #
# ---------------------------------------------------------------------------- #
class Mouse:
    """Represents the state of the micromouse."""
    def __init__(self):
        # Start at the center of the bottom-left cell (0,0).
        # In a half-cell grid (1 unit = 8cm), this is (1,1).
        self.x = 1
        self.y = 1
        self.orientation = 0  # 0:N, 45:NE, 90:E, ..., 315:NW
        self.momentum = 0

# ---------------------------------------------------------------------------- #
#                                   Maze Class                                 #
# ---------------------------------------------------------------------------- #
class Maze:
    """Represents the maze environment and rules."""
    def __init__(self, width=16, height=16):
        # Maze dimensions in full cells
        self.width = width
        self.height = height

    def is_valid_position(self, x, y):
        """Checks if a position is within the maze boundaries (in half-cells)."""
        # The grid spans from 0 to width*2 and 0 to height*2
        return 0 <= x <= self.width * 2 and 0 <= y <= self.height * 2

    def is_in_goal_interior(self, x, y):
        """
        Checks if the mouse is entirely within the goal's interior.
        Goal is the central 2x2 block (cells (7,7), (7,8), (8,7), (8,8)).
        In half-cell coordinates, the interior is the central 2x2 half-cell square.
        This corresponds to x values of 15, 16 and y values of 15, 16.
        A stop must be at a cell center (odd, odd), so (15,15) is the key goal center.
        """
        return (14 < x < 18) and (14 < y < 18)

# ---------------------------------------------------------------------------- #
#                                Movement Class                                #
# ---------------------------------------------------------------------------- #
class Movement:
    """
    Acts as the physics engine for the mouse. It parses instructions,
    calculates time, updates state, and checks for crashes.
    """
    def __init__(self, mouse, maze):
        self.mouse = mouse
        self.maze = maze
        self.BASE_TIMES = {
            "inplace_turn": 200,
            "default_rest": 200,
            "half_step_cardinal": 500,
            "half_step_intercardinal": 600,
            "corner_tight": 700,
            "corner_wide": 1400,
        }
        self.REDUCTION_MAP = {
            0.0: 0.00, 0.5: 0.10, 1.0: 0.20, 1.5: 0.275, 2.0: 0.35,
            2.5: 0.40, 3.0: 0.45, 3.5: 0.475, 4.0: 0.50
        }
        # Delta (dx, dy) for each 45-degree orientation for a half-step move
        self.ORIENTATION_DELTAS = {
            0:   (0, 1), 45:  (1, 1), 90:  (1, 0), 135: (1, -1),
            180: (0, -1), 225: (-1, -1), 270: (-1, 0), 315: (-1, 1)
        }

    def _calculate_time(self, base_time, m_in, m_out):
        """Calculates the time for a move based on momentum reduction."""
        m_eff = (abs(m_in) + abs(m_out)) / 2.0
        # Interpolate if m_eff is not in the map (though the rules imply it will be)
        reduction = self.REDUCTION_MAP.get(m_eff, 0.0)
        return round(base_time * (1 - reduction))

    def execute_instruction(self, instruction):
        """
        Executes a single instruction.
        Returns: (time_ms, crashed, crash_reason)
        """
        m_in = self.mouse.momentum

        # --- 1. Check for Corner Turns ---
        is_corner_turn = ('T' in instruction or 'W' in instruction)
        if is_corner_turn:
            return self._execute_corner_turn(instruction, m_in)

        # --- 2. Parse Longitudinal and Moving Rotations ---
        move_part = instruction
        rotation_part = None
        if len(instruction) > 1 and instruction[-1] in ['L', 'R']:
             if instruction[:-1] in ["F0", "F1", "F2", "V0", "V1", "V2", "BB"]:
                 move_part = instruction[:-1]
                 rotation_part = instruction[-1]

        # --- 3. In-place Rotations ---
        if move_part in ['L', 'R'] and not rotation_part:
            if m_in != 0:
                return 0, True, "In-place rotation requires momentum 0"
            self.mouse.orientation = (self.mouse.orientation - 45 if move_part == 'L' else self.mouse.orientation + 45) % 360
            return self.BASE_TIMES["inplace_turn"], False, None

        # --- 4. Longitudinal Moves ---
        # Determine momentum change
        m_out = m_in
        if move_part.startswith('F'):
            if m_in < 0: return 0, True, "Cannot use F move with reverse momentum"
            if move_part == 'F2': m_out = min(4, m_in + 1)
            elif move_part == 'F0': m_out = max(0, m_in - 1)
        elif move_part.startswith('V'):
            if m_in > 0: return 0, True, "Cannot use V move with forward momentum"
            if move_part == 'V2': m_out = max(-4, m_in - 1)
            elif move_part == 'V0': m_out = min(0, m_in + 1)
        elif move_part == 'BB':
            if m_in > 0: m_out = max(0, m_in - 2)
            elif m_in < 0: m_out = min(0, m_in + 2)
            else: return self.BASE_TIMES["default_rest"], False, None # BB at rest

        # Crash if reversing direction without stopping
        if (m_in > 0 and m_out < 0) or (m_in < 0 and m_out > 0):
            return 0, True, "Cannot accelerate in opposite direction without stopping"

        # Calculate time and update position
        is_intercardinal = self.mouse.orientation % 90 != 0
        base_time = self.BASE_TIMES["half_step_intercardinal"] if is_intercardinal else self.BASE_TIMES["half_step_cardinal"]
        time_ms = self._calculate_time(base_time, m_in, m_out)

        dx, dy = self.ORIENTATION_DELTAS.get(self.mouse.orientation, (0,0))
        self.mouse.x += dx
        self.mouse.y += dy
        self.mouse.momentum = m_out

        if not self.maze.is_valid_position(self.mouse.x, self.mouse.y):
            return time_ms, True, "Moved out of maze boundaries"

        # Handle rotation part of a moving rotation
        if rotation_part:
            m_eff = (abs(m_in) + abs(m_out)) / 2.0
            if m_eff > 1:
                return time_ms, True, f"Moving rotation m_eff ({m_eff}) exceeds 1"
            self.mouse.orientation = (self.mouse.orientation - 45 if rotation_part == 'L' else self.mouse.orientation + 45) % 360

        return time_ms, False, None

    def _execute_corner_turn(self, instruction, m_in):
        """Handles the specific logic for corner turns."""
        # Constraint: Must start facing a cardinal direction
        if self.mouse.orientation % 90 != 0:
            return 0, True, "Corner turns must start from a cardinal direction"

        # Parse token
        move_token = instruction[0:2]
        turn_dir = instruction[2] # L or R
        radius = instruction[3] # T or W
        end_rot = instruction[4] if len(instruction) > 4 else None

        # Determine momentum change
        m_out = m_in
        if move_token.startswith('F'):
            if m_in < 0: return 0, True, "Cannot use F move with reverse momentum"
            if move_token == 'F2': m_out = min(4, m_in + 1)
            elif move_token == 'F0': m_out = max(0, m_in - 1)
        # Add V-move logic if corner turns with V are allowed by rules
        # (Original rules only listed F, but we'll assume they are analogous)

        m_eff = (abs(m_in) + abs(m_out)) / 2.0

        # Constraint: Check effective momentum
        if radius == 'T' and m_eff > 1:
            return 0, True, f"Tight corner turn m_eff ({m_eff}) exceeds 1"
        if radius == 'W' and m_eff > 2:
            return 0, True, f"Wide corner turn m_eff ({m_eff}) exceeds 2"

        # Calculate time
        base_time = self.BASE_TIMES["corner_tight"] if radius == 'T' else self.BASE_TIMES["corner_wide"]
        time_ms = self._calculate_time(base_time, m_in, m_out)

        # Update position and orientation
        self.mouse.momentum = m_out
        
        # A 90-degree turn moves a certain number of half-steps fwd and side
        # Tight turn = 1 fwd, 1 side. Wide turn = 2 fwd, 2 side.
        steps = 1 if radius == 'T' else 2
        
        # Get forward and sideways vectors
        fwd_dx, fwd_dy = self.ORIENTATION_DELTAS[self.mouse.orientation]
        side_orientation = (self.mouse.orientation - 90 if turn_dir == 'L' else self.mouse.orientation + 90) % 360
        side_dx, side_dy = self.ORIENTATION_DELTAS[side_orientation]
        
        self.mouse.x += (fwd_dx + side_dx) * steps
        self.mouse.y += (fwd_dy + side_dy) * steps
        
        # Update orientation
        self.mouse.orientation = (self.mouse.orientation - 90 if turn_dir == 'L' else self.mouse.orientation + 90) % 360
        if end_rot:
             self.mouse.orientation = (self.mouse.orientation - 45 if end_rot == 'L' else self.mouse.orientation + 45) % 360
        
        return time_ms, False, None

# ---------------------------------------------------------------------------- #
#                                Flask Application                             #
# ---------------------------------------------------------------------------- #
games = {} # In-memory store for game states

@app.route('/micro-mouse', methods=['POST'])
def micro_mouse():
    data = request.json
    game_uuid = data.get("game_uuid")

    if not game_uuid:
        return jsonify({"error": "game_uuid is required"}), 400

    if game_uuid not in games:
        games[game_uuid] = {"mouse": Mouse(), "maze": Maze()}

    state = games[game_uuid]
    mouse = state["mouse"]
    maze = state["maze"]
    movement = Movement(mouse, maze)

    # Use state from the request to update our simulation
    # In a real scenario, you'd sync your server state with the incoming data
    # For this example, we manage state internally after initialization
    total_time_ms = data.get("total_time_ms", 0)
    best_time_ms = data.get("best_time_ms")
    run_time_ms = data.get("run_time_ms", 0)
    run = data.get("run", 0)

    # Check for end flag
    if data.get("end"):
        return jsonify({"message": f"Challenge ended by user. Final score based on best_time: {best_time_ms}"})

    # Add thinking time for a valid, non-empty request
    instructions = data.get("instructions", [])
    if instructions:
        total_time_ms += 50
        is_at_start_center = (mouse.x == 1 and mouse.y == 1)
        if not is_at_start_center:
            run_time_ms += 50

    # Execute instructions
    for instruction in instructions:
        time_taken, crashed, reason = movement.execute_instruction(instruction)
        
        is_at_start_center = (mouse.x == 1 and mouse.y == 1)

        # Update timers
        total_time_ms += time_taken
        if not is_at_start_center or mouse.momentum != 0:
             run_time_ms += time_taken
        
        if crashed:
            del games[game_uuid] # Clean up crashed game
            return jsonify({"error": "Mouse crashed!", "reason": reason, "instruction": instruction}), 400

    # Check for game events after moves
    if maze.is_in_goal_interior(mouse.x, mouse.y) and mouse.momentum == 0:
        if best_time_ms is None or run_time_ms < best_time_ms:
            best_time_ms = run_time_ms

    if mouse.x == 1 and mouse.y == 1 and mouse.momentum == 0:
        run += 1
        run_time_ms = 0

    # ------------------------------------------------------------------------ #
    # TODO: INSERT YOUR MAZE-SOLVING ALGORITHM HERE
    # Based on sensor_data and current mouse state, decide the next moves.
    # The response below is a placeholder.
    # ------------------------------------------------------------------------ #
    response_instructions = ["F2", "F2", "F1", "F1RT"]

    return jsonify({
        "instructions": response_instructions,
        "end": False
    })

if __name__ == '__main__':
    app.run(debug=True, port=5001)