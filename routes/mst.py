from flask import Flask, request, jsonify
from routes import app
import cv2
import numpy as np
import base64
from PIL import Image
import io
import pytesseract
from scipy.spatial.distance import euclidean
import heapq
from collections import defaultdict


@app.route("/mst-calculation", methods=["POST"])
def mst_calculation():
    data = request.get_json()
    results = []
    
    for test_case in data:
        image_data = base64.b64decode(test_case["image"])
        image = Image.open(io.BytesIO(image_data))
        
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        nodes, edges = extract_graph_from_image(opencv_image)
        
        mst_weight = calculate_mst_weight(nodes, edges)
        
        results.append({"value": mst_weight})
    
    return jsonify(results)

def extract_graph_from_image(image):
    """Extract nodes and edges from the graph image"""
    # Convert to different color spaces for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Detect nodes (black circles)
    nodes = detect_nodes(gray)
    
    # Detect edges and their weights
    edges = detect_edges_and_weights(image, gray, nodes)
    
    return nodes, edges

def detect_nodes(gray_image):
    """Detect black circular nodes in the image"""
    nodes = []
    
    # Use HoughCircles to detect circular shapes
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=5,
        maxRadius=50
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Check if the circle is dark (black node)
            center_intensity = gray_image[y, x]
            if center_intensity < 100:  # Dark threshold
                nodes.append((x, y))
    
    # Alternative method: contour detection for black regions
    if len(nodes) < 3:  # Fallback if HoughCircles doesn't work well
        # Create binary mask for dark regions
        _, binary = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Filter by area and circularity
            area = cv2.contourArea(contour)
            if 50 < area < 2000:
                # Check circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.4:  # Reasonably circular
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            nodes.append((cx, cy))
    
    return nodes

def detect_edges_and_weights(image, gray, nodes):
    """Detect edges between nodes and extract their weights"""
    edges = []
    height, width = gray.shape
    
    # Create edge detection mask
    edges_mask = cv2.Canny(gray, 50, 150)
    
    # For each pair of nodes, check if there's an edge
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            node1, node2 = nodes[i], nodes[j]
            
            # Check if there's a line between these nodes
            if has_edge_between_nodes(edges_mask, node1, node2):
                # Extract weight along the edge
                weight = extract_weight_on_edge(image, node1, node2)
                if weight > 0:
                    edges.append((i, j, weight))
    
    return edges

def has_edge_between_nodes(edges_mask, node1, node2):
    """Check if there's an edge between two nodes using line detection"""
    x1, y1 = node1
    x2, y2 = node2
    
    # Sample points along the line between nodes
    num_samples = max(int(euclidean(node1, node2) / 5), 10)
    
    edge_pixels = 0
    for i in range(1, num_samples):
        t = i / num_samples
        x = int(x1 + t * (x2 - x1))
        y = int(y1 + t * (y2 - y1))
        
        if 0 <= x < edges_mask.shape[1] and 0 <= y < edges_mask.shape[0]:
            if edges_mask[y, x] > 0:
                edge_pixels += 1
    
    # If enough pixels along the line are edge pixels, consider it an edge
    return edge_pixels > num_samples * 0.1

def extract_weight_on_edge(image, node1, node2):
    """Extract the numeric weight displayed on the edge between two nodes"""
    x1, y1 = node1
    x2, y2 = node2
    
    # Find midpoint of the edge
    mid_x = (x1 + x2) // 2
    mid_y = (y1 + y2) // 2
    
    # Extract region around midpoint for OCR
    region_size = 40
    x_start = max(0, mid_x - region_size // 2)
    x_end = min(image.shape[1], mid_x + region_size // 2)
    y_start = max(0, mid_y - region_size // 2)
    y_end = min(image.shape[0], mid_y + region_size // 2)
    
    roi = image[y_start:y_end, x_start:x_end]
    
    # Try multiple preprocessing approaches for OCR
    weights = []
    
    # Method 1: Direct OCR on color image
    weight = ocr_extract_number(roi)
    if weight > 0:
        weights.append(weight)
    
    # Method 2: Convert to grayscale and threshold
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh_roi = cv2.threshold(gray_roi, 127, 255, cv2.THRESH_BINARY)
    weight = ocr_extract_number(thresh_roi)
    if weight > 0:
        weights.append(weight)
    
    # Method 3: Try with inverted colors
    inv_thresh = cv2.bitwise_not(thresh_roi)
    weight = ocr_extract_number(inv_thresh)
    if weight > 0:
        weights.append(weight)
    
    # Method 4: Enhanced contrast
    enhanced = cv2.convertScaleAbs(gray_roi, alpha=2.0, beta=0)
    weight = ocr_extract_number(enhanced)
    if weight > 0:
        weights.append(weight)
    
    # Return most common weight or first valid one
    if weights:
        return max(set(weights), key=weights.count)
    
    # Fallback: try to extract from a larger region
    larger_size = 60
    x_start = max(0, mid_x - larger_size // 2)
    x_end = min(image.shape[1], mid_x + larger_size // 2)
    y_start = max(0, mid_y - larger_size // 2)
    y_end = min(image.shape[0], mid_y + larger_size // 2)
    
    larger_roi = image[y_start:y_end, x_start:x_end]
    return ocr_extract_number(larger_roi)

def ocr_extract_number(roi):
    """Use OCR to extract number from region of interest"""
    try:
        # Configure tesseract for digits only
        config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(roi, config=config).strip()
        
        # Extract first number found
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            return int(numbers[0])
    except:
        pass
    
    return 0

def calculate_mst_weight(nodes, edges):
    """Calculate the total weight of the minimum spanning tree using Kruskal's algorithm"""
    if not nodes or not edges:
        return 0
    
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])
    
    # Initialize Union-Find structure
    parent = list(range(len(nodes)))
    rank = [0] * len(nodes)
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True
    
    # Kruskal's algorithm
    mst_weight = 0
    edges_added = 0
    
    for u, v, weight in edges:
        if union(u, v):
            mst_weight += weight
            edges_added += 1
            if edges_added == len(nodes) - 1:
                break
    
    return mst_weight

# Alternative MST implementation using Prim's algorithm
def calculate_mst_weight_prim(nodes, edges):
    """Calculate MST weight using Prim's algorithm (alternative implementation)"""
    if not nodes or not edges:
        return 0
    
    n = len(nodes)
    if n == 1:
        return 0
    
    # Build adjacency list
    graph = defaultdict(list)
    for u, v, weight in edges:
        graph[u].append((v, weight))
        graph[v].append((u, weight))
    
    # Prim's algorithm
    visited = [False] * n
    min_heap = [(0, 0)]  # (weight, node)
    mst_weight = 0
    
    while min_heap:
        weight, u = heapq.heappop(min_heap)
        
        if visited[u]:
            continue
            
        visited[u] = True
        mst_weight += weight
        
        for v, edge_weight in graph[u]:
            if not visited[v]:
                heapq.heappush(min_heap, (edge_weight, v))
    
    return mst_weight

if __name__ == "__main__":
    app.run(port=3000, debug=True)