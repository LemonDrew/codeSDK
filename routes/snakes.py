from flask import Flask, request, Response
import xml.etree.ElementTree as ET
from typing import List, Tuple, Set
import random

from routes import app

class SnakesLaddersGame:
    def __init__(self, svg_content: str):
        self.svg_content = svg_content
        self.width = 0
        self.height = 0
        self.board_size = 0
        self.jumps = {}  # start_square -> end_square
        self.parse_svg()
        
    def parse_svg(self):
        """Parse the SVG to extract board dimensions and jumps"""
        try:
            root = ET.fromstring(self.svg_content)
            
            # Extract viewBox to get dimensions
            viewbox = root.get('viewBox', '0 0 512 512')
            parts = viewbox.split()
            svg_width = int(parts[2])
            svg_height = int(parts[3])
            
            # Calculate board dimensions (each square is 32x32)
            self.width = svg_width // 32
            self.height = svg_height // 32
            self.board_size = self.width * self.height
            
            # Parse jumps from line elements
            for line in root.findall('.//{http://www.w3.org/2000/svg}line'):
                x1 = float(line.get('x1', 0))
                y1 = float(line.get('y1', 0))
                x2 = float(line.get('x2', 0))
                y2 = float(line.get('y2', 0))
                
                start_square = self.coord_to_square(x1, y1)
                end_square = self.coord_to_square(x2, y2)
                
                if start_square and end_square:
                    self.jumps[start_square] = end_square
                    
        except Exception as e:
            # Default fallback
            self.width = 16
            self.height = 16
            self.board_size = 256
            
    def coord_to_square(self, x: float, y: float) -> int:
        """Convert SVG coordinates to square number"""
        try:
            # Convert coordinates to grid position
            col = int(x // 32)
            row = int(y // 32)
            
            # Handle boustrophedon numbering (snake pattern)
            if row % 2 == 0:  # Even rows go left to right
                square = (self.height - 1 - row) * self.width + col + 1
            else:  # Odd rows go right to left
                square = (self.height - 1 - row) * self.width + (self.width - 1 - col) + 1
                
            return square if 1 <= square <= self.board_size else None
        except:
            return None
    
    def move_player(self, position: int, roll: int, is_power_die: bool) -> int:
        """Move player and handle overshooting"""
        if is_power_die:
            move_distance = 2 ** roll  # Power of 2: 2, 4, 8, 16, 32, 64
        else:
            move_distance = roll
            
        new_position = position + move_distance
        
        # Handle overshooting
        if new_position > self.board_size:
            overshoot = new_position - self.board_size
            new_position = self.board_size - overshoot
            
        # Apply jumps (snakes/ladders)
        if new_position in self.jumps:
            new_position = self.jumps[new_position]
            
        return max(0, new_position)
    
    def simulate_game(self, rolls: List[int]) -> Tuple[bool, Set[int]]:
        """Simulate the game with given rolls and return if player 2 wins and squares visited"""
        player1_pos = 0
        player2_pos = 0
        player1_power = False
        player2_power = False
        visited_squares = set()
        
        roll_index = 0
        current_player = 1  # Start with player 1
        
        while roll_index < len(rolls):
            roll = rolls[roll_index]
            
            if current_player == 1:
                # Player 1 turn
                old_pos = player1_pos
                player1_pos = self.move_player(player1_pos, roll, player1_power)
                visited_squares.add(player1_pos)
                
                # Update power state
                if not player1_power and roll == 6:
                    player1_power = True
                elif player1_power and roll == 1:
                    player1_power = False
                    
                if player1_pos >= self.board_size:
                    return False, visited_squares  # Player 1 wins
                    
            else:
                # Player 2 turn
                old_pos = player2_pos
                player2_pos = self.move_player(player2_pos, roll, player2_power)
                visited_squares.add(player2_pos)
                
                # Update power state
                if not player2_power and roll == 6:
                    player2_power = True
                elif player2_power and roll == 1:
                    player2_power = False
                    
                if player2_pos >= self.board_size:
                    return True, visited_squares  # Player 2 wins
            
            # Switch players
            current_player = 2 if current_player == 1 else 1
            roll_index += 1
            
        return False, visited_squares  # Game didn't end
    
    def generate_solution(self) -> str:
        """Generate a sequence of rolls that makes player 2 win"""
        best_rolls = []
        best_coverage = 0
        
        # Try multiple strategies
        for attempt in range(100):
            rolls = self.generate_strategic_rolls()
            player2_wins, visited = self.simulate_game(rolls)
            
            if player2_wins:
                coverage = len(visited) / self.board_size
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_rolls = rolls
                    
                # If we get decent coverage, use it
                if coverage >= 0.3:
                    break
        
        # Fallback: simple alternating strategy
        if not best_rolls:
            best_rolls = self.simple_strategy()
            
        return ''.join(map(str, best_rolls))
    
    def generate_strategic_rolls(self) -> List[int]:
        """Generate strategic dice rolls"""
        rolls = []
        target_length = min(200, self.board_size * 2)
        
        # Strategy: Give player 1 some progress, then help player 2 win
        player1_turns = random.randint(5, 15)
        
        # Player 1 gets moderate rolls
        for i in range(player1_turns * 2):
            if i % 2 == 0:  # Player 1's turn
                rolls.append(random.randint(1, 4))  # Conservative rolls
            else:  # Player 2's turn
                rolls.append(random.randint(1, 3))  # Very conservative
                
        # Now help player 2 with better rolls
        while len(rolls) < target_length:
            if len(rolls) % 2 == 0:  # Player 1's turn
                rolls.append(random.randint(1, 3))  # Keep player 1 slow
            else:  # Player 2's turn
                if random.random() < 0.3:  # 30% chance of 6 to get power die
                    rolls.append(6)
                else:
                    rolls.append(random.randint(3, 6))  # Better rolls for player 2
                    
        return rolls
    
    def simple_strategy(self) -> List[int]:
        """Simple fallback strategy"""
        rolls = []
        
        # Give player 1 small rolls, player 2 better rolls
        for i in range(100):
            if i % 2 == 0:  # Player 1
                rolls.append(random.randint(1, 3))
            else:  # Player 2
                if i > 20 and random.random() < 0.4:
                    rolls.append(6)  # Power up player 2
                else:
                    rolls.append(random.randint(4, 6))
                    
        return rolls

@app.route('/slpu', methods=['POST'])
def snakes():
    try:
        # Get SVG content from request
        svg_content = request.get_data(as_text=True)
        
        # Create game instance and solve
        game = SnakesLaddersGame(svg_content)
        solution = game.generate_solution()
        
        # Return solution as SVG
        response_svg = f'<svg xmlns="http://www.w3.org/2000/svg"><text>{solution}</text></svg>'
        
        return Response(response_svg, mimetype='image/svg+xml')
        
    except Exception as e:
        # Fallback solution
        fallback = '<svg xmlns="http://www.w3.org/2000/svg"><text>123456123456</text></svg>'
        return Response(fallback, mimetype='image/svg+xml')

