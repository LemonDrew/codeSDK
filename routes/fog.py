from flask import Flask, request, jsonify
import json

from routes import app

class MazeSolver:
    def __init__(self):
        self.game_state = {}
    
    def initialize_game(self, test_case, challenger_id, game_id):
        """Initialize a new game state"""
        self.game_state = {
            'challenger_id': challenger_id,
            'game_id': game_id,
            'grid_size': test_case['length_of_grid'],
            'num_walls': test_case['num_of_walls'],
            'crows': {crow['id']: {'x': crow['x'], 'y': crow['y']} for crow in test_case['crows']},
            'discovered_walls': set(),
            'scanned_positions': set(),
            'current_crow_index': 0,
            'crow_ids': [crow['id'] for crow in test_case['crows']],
            'scan_positions': [],
            'current_scan_index': 0
        }
        
        # Generate all positions we want to scan
        # We'll scan every position that can give us maximum coverage
        # For a 5x5 scan area, we want to place the crow at positions where
        # the scan covers areas we haven't seen yet
        self.generate_scan_positions()
    
    def generate_scan_positions(self):
        """Generate positions where we should place crows to scan"""
        grid_size = self.game_state['grid_size']
        positions = []
        
        # Scan in a grid pattern with spacing of 3 (since scan radius is 2)
        # This ensures we cover everything with minimal overlap
        for y in range(2, grid_size - 2, 3):
            for x in range(2, grid_size - 2, 3):
                positions.append((x, y))
        
        # Add edge positions to make sure we cover the entire grid
        # Top and bottom edges
        for x in range(2, grid_size - 2, 3):
            if 0 not in [pos[1] for pos in positions if pos[0] == x]:
                positions.append((x, 2))
            if grid_size - 3 not in [pos[1] for pos in positions if pos[0] == x]:
                positions.append((x, min(grid_size - 3, grid_size - 1)))
        
        # Left and right edges
        for y in range(2, grid_size - 2, 3):
            if 0 not in [pos[0] for pos in positions if pos[1] == y]:
                positions.append((2, y))
            if grid_size - 3 not in [pos[0] for pos in positions if pos[1] == y]:
                positions.append((min(grid_size - 3, grid_size - 1), y))
        
        self.game_state['scan_positions'] = positions
    
    def get_current_crow_id(self):
        """Get the ID of the currently active crow"""
        crow_ids = self.game_state['crow_ids']
        current_index = self.game_state['current_crow_index']
        return crow_ids[current_index % len(crow_ids)]
    
    def get_next_action(self):
        """Determine the next action to take"""
        if self.game_state['current_scan_index'] >= len(self.game_state['scan_positions']):
            # We've covered all planned scan positions, submit what we found
            return self.submit_walls()
        
        current_crow_id = self.get_current_crow_id()
        current_crow = self.game_state['crows'][current_crow_id]
        target_pos = self.game_state['scan_positions'][self.game_state['current_scan_index']]
        
        # Check if crow is at target position
        if current_crow['x'] == target_pos[0] and current_crow['y'] == target_pos[1]:
            # Scan at this position
            return {
                'challenger_id': self.game_state['challenger_id'],
                'game_id': self.game_state['game_id'],
                'crow_id': current_crow_id,
                'action_type': 'scan'
            }
        else:
            # Move towards target position
            direction = self.get_direction_to_target(current_crow, target_pos)
            return {
                'challenger_id': self.game_state['challenger_id'],
                'game_id': self.game_state['game_id'],
                'crow_id': current_crow_id,
                'action_type': 'move',
                'direction': direction
            }
    
    def get_direction_to_target(self, crow_pos, target_pos):
        """Get the direction to move towards target position"""
        dx = target_pos[0] - crow_pos['x']
        dy = target_pos[1] - crow_pos['y']
        
        # Prioritize moving in the direction with larger difference
        if abs(dx) >= abs(dy):
            return 'E' if dx > 0 else 'W'
        else:
            return 'S' if dy > 0 else 'N'
    
    def process_move_result(self, previous_action):
        """Update crow position after a move"""
        crow_id = previous_action['crow_id']
        new_pos = previous_action['move_result']
        self.game_state['crows'][crow_id]['x'] = new_pos[0]
        self.game_state['crows'][crow_id]['y'] = new_pos[1]
    
    def process_scan_result(self, previous_action):
        """Process the result of a scan and extract wall positions"""
        crow_id = previous_action['crow_id']
        scan_result = previous_action['scan_result']
        crow_pos = self.game_state['crows'][crow_id]
        
        # The scan result is 5x5 with the crow at center [2][2]
        # Convert relative positions to absolute grid positions
        for i in range(5):
            for j in range(5):
                if scan_result[i][j] == 'W':
                    # Calculate absolute position
                    abs_x = crow_pos['x'] + (j - 2)
                    abs_y = crow_pos['y'] + (i - 2)
                    self.game_state['discovered_walls'].add(f"{abs_x}-{abs_y}")
        
        # Mark this position as scanned and move to next scan position
        current_pos = self.game_state['scan_positions'][self.game_state['current_scan_index']]
        self.game_state['scanned_positions'].add(current_pos)
        self.game_state['current_scan_index'] += 1
        
        # Move to next crow for variety (optional, helps if one crow gets stuck)
        self.game_state['current_crow_index'] += 1
    
    def submit_walls(self):
        """Submit all discovered walls"""
        return {
            'challenger_id': self.game_state['challenger_id'],
            'game_id': self.game_state['game_id'],
            'action_type': 'submit',
            'submission': list(self.game_state['discovered_walls'])
        }

# Global solver instance
solver = MazeSolver()

@app.route('/fog-of-wall', methods=['POST'])
def fog():
    try:
        data = request.get_json()
        
        challenger_id = data['challenger_id']
        game_id = data['game_id']
        
        # Check if this is a new test case
        if 'test_case' in data:
            # Initialize new game
            solver.initialize_game(data['test_case'], challenger_id, game_id)
            response = solver.get_next_action()
        else:
            # Process previous action result
            previous_action = data['previous_action']
            
            if previous_action['your_action'] == 'move':
                solver.process_move_result(previous_action)
            elif previous_action['your_action'] == 'scan':
                solver.process_scan_result(previous_action)
            
            response = solver.get_next_action()
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)