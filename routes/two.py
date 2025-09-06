from flask import Flask, request, jsonify
import random

from routes import app

def add_random_tile(grid):
    """Add a 2 or 4 tile randomly in an empty cell"""
    empty = [(r, c) for r in range(4) for c in range(4) if grid[r][c] is None]
    if not empty:
        return grid
    r, c = random.choice(empty)
    grid[r][c] = 2 if random.random() < 0.9 else 4
    return grid

def compress_and_merge(line):
    """Slide tiles in one row/col, merging equal ones"""
    new_line = [x for x in line if x is not None]
    merged = []
    i = 0
    while i < len(new_line):
        if i + 1 < len(new_line) and new_line[i] == new_line[i+1]:
            merged.append(new_line[i] * 2)
            i += 2
        else:
            merged.append(new_line[i])
            i += 1
    while len(merged) < 4:
        merged.append(None)
    return merged

def move_grid(grid, direction):
    """Perform a move in the given direction"""
    new_grid = [[grid[r][c] for c in range(4)] for r in range(4)]
    if direction == "LEFT":
        for r in range(4):
            new_grid[r] = compress_and_merge(new_grid[r])
    elif direction == "RIGHT":
        for r in range(4):
            row = list(reversed(new_grid[r]))
            row = compress_and_merge(row)
            new_grid[r] = list(reversed(row))
    elif direction == "UP":
        for c in range(4):
            col = [new_grid[r][c] for r in range(4)]
            col = compress_and_merge(col)
            for r in range(4):
                new_grid[r][c] = col[r]
    elif direction == "DOWN":
        for c in range(4):
            col = [new_grid[r][c] for r in range(4)]
            col = list(reversed(col))
            col = compress_and_merge(col)
            col = list(reversed(col))
            for r in range(4):
                new_grid[r][c] = col[r]
    return new_grid

def check_endgame(grid):
    """Return 'win', 'lose', or None"""
    # Win check
    for row in grid:
        if 2048 in row:
            return "win"
    # Empty cell check
    for row in grid:
        if None in row:
            return None
    # Move possibility check
    for r in range(4):
        for c in range(4):
            if r < 3 and grid[r][c] == grid[r+1][c]:
                return None
            if c < 3 and grid[r][c] == grid[r][c+1]:
                return None
    return "lose"

@app.route("/2048", methods=["POST"])
def play():
    data = request.get_json(force=True)
    grid = data.get("grid")
    direction = data.get("mergeDirection")

    # Apply move
    next_grid = move_grid(grid, direction)

    # Add random tile
    next_grid = add_random_tile(next_grid)

    # Check endgame
    endgame = check_endgame(next_grid)

    return jsonify({
        "nextGrid": next_grid,
        "endGame": endgame
    })
