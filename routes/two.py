from flask import request, jsonify
import random

from routes import app

# --- Helpers ---

def add_random_tile(grid):
    empty = [(r, c) for r in range(4) for c in range(4) if grid[r][c] is None]
    if not empty:
        return
    r, c = random.choice(empty)
    grid[r][c] = 4 if random.random() < 0.1 else 2

def compress_and_merge(row):
    """Slide row left and merge"""
    row = [x for x in row if x is not None]
    result = []
    skip = False
    for i in range(len(row)):
        if skip:
            skip = False
            continue
        if i + 1 < len(row) and row[i] == row[i+1]:
            result.append(row[i] * 2)
            skip = True
        else:
            result.append(row[i])
    result += [None] * (4 - len(result))
    return result

def move_grid(grid, direction):
    new_grid = [[grid[r][c] for c in range(4)] for r in range(4)]
    if direction == "LEFT":
        for r in range(4):
            new_grid[r] = compress_and_merge(new_grid[r])
    elif direction == "RIGHT":
        for r in range(4):
            row = compress_and_merge(new_grid[r][::-1])
            new_grid[r] = row[::-1]
    elif direction == "UP":
        for c in range(4):
            col = compress_and_merge([new_grid[r][c] for r in range(4)])
            for r in range(4):
                new_grid[r][c] = col[r]
    elif direction == "DOWN":
        for c in range(4):
            col = compress_and_merge([new_grid[r][c] for r in range(4)][::-1])
            col = col[::-1]
            for r in range(4):
                new_grid[r][c] = col[r]
    return new_grid

def check_endgame(grid):
    # win condition
    if any(cell == 2048 for row in grid for cell in row if cell):
        return "win"
    # empty cell exists
    if any(cell is None for row in grid for cell in row):
        return None
    # check possible merges
    for r in range(4):
        for c in range(4):
            if r < 3 and grid[r][c] == grid[r+1][c]:
                return None
            if c < 3 and grid[r][c] == grid[r][c+1]:
                return None
    return "lose"

# --- Endpoint ---

@app.route("/2048", methods=["POST"])
def play_2048():
    data = request.get_json()
    grid = data.get("grid")
    direction = data.get("mergeDirection")

    new_grid = move_grid(grid, direction)
    if new_grid != grid:  # only add random tile if grid changed
        add_random_tile(new_grid)

    endGame = check_endgame(new_grid)
    return jsonify({"nextGrid": new_grid, "endGame": endGame})
