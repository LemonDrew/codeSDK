# routes.py
from flask import Flask, request, jsonify
import random

from routes import app  
app.config['DEBUG'] = True  # Ensure debug logs are shown

# Helper functions for 2048 logic
def slide_and_merge(row):
    """Slide non-empty tiles to the left and merge equal tiles."""
    new_row = [x for x in row if x is not None]
    merged = []
    skip = False
    for i in range(len(new_row)):
        if skip:
            skip = False
            continue
        if i + 1 < len(new_row) and new_row[i] == new_row[i + 1]:
            merged.append(new_row[i] * 2)
            skip = True
        else:
            merged.append(new_row[i])
    while len(merged) < 4:
        merged.append(None)
    app.logger.debug(f"Row before merge: {row} -> after merge: {merged}")
    return merged

def transpose(grid):
    t = [list(row) for row in zip(*grid)]
    app.logger.debug(f"Transposed grid: {t}")
    return t

def reverse(grid):
    r = [row[::-1] for row in grid]
    app.logger.debug(f"Reversed grid: {r}")
    return r

def move_grid(grid, direction):
    app.logger.debug(f"Moving grid {direction}")
    if direction == "UP":
        grid = transpose(grid)
        grid = [slide_and_merge(row) for row in grid]
        grid = transpose(grid)
    elif direction == "DOWN":
        grid = transpose(grid)
        grid = [slide_and_merge(row[::-1])[::-1] for row in grid]
        grid = transpose(grid)
    elif direction == "LEFT":
        grid = [slide_and_merge(row) for row in grid]
    elif direction == "RIGHT":
        grid = [slide_and_merge(row[::-1])[::-1] for row in grid]
    app.logger.debug(f"Grid after move {direction}: {grid}")
    return grid

def add_random_tile(grid):
    empty = [(r, c) for r in range(4) for c in range(4) if grid[r][c] is None]
    if empty:
        r, c = random.choice(empty)
        grid[r][c] = 2 if random.random() < 0.9 else 4
        app.logger.debug(f"Added tile {grid[r][c]} at position ({r},{c})")
    return grid

def check_endgame(grid):
    for r in range(4):
        for c in range(4):
            if grid[r][c] is None:
                return None
            if r < 3 and grid[r][c] == grid[r + 1][c]:
                return None
            if c < 3 and grid[r][c] == grid[r][c + 1]:
                return None
    for row in grid:
        if 2048 in row:
            return "win"
    return "lose"

@app.route("/2048", methods=["POST"])
def play():
    try:
        data = request.get_json(force=True)
        grid = data.get("grid")
        direction = data.get("mergeDirection")
        app.logger.debug(f"Received grid: {grid} with direction: {direction}")

        # Move tiles
        new_grid = move_grid(grid, direction)
        # Add a new random tile
        new_grid = add_random_tile(new_grid)
        # Check endgame
        endgame = check_endgame(new_grid)
        app.logger.debug(f"New grid: {new_grid}, endgame status: {endgame}")

        return jsonify({"nextGrid": new_grid, "endGame": endgame})
    except Exception as e:
        app.logger.error(f"Error processing move: {e}")
        return jsonify({"nextGrid": grid, "endGame": None})
