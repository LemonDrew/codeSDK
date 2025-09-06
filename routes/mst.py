import base64
import io
import cv2
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import pytesseract
from typing import List, Tuple, Dict, Set
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class GraphExtractor:
    """Extracts graph structure from images"""
    
    def __init__(self):
        self.nodes = []
        self.edges = []
        
    def detect_nodes(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """Detect circular nodes (black circles) in the image"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Use HoughCircles to detect circular nodes
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=40
        )
        
        nodes = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                nodes.append((int(x), int(y)))
        
        return nodes
    
    def detect_edges_and_weights(self, img: np.ndarray, nodes: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
        """Detect edges between nodes and extract their weights"""
        edges = []
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Create a mask to remove nodes from the image (to better detect edges)
        mask = np.ones(gray.shape, dtype=np.uint8) * 255
        for x, y in nodes:
            cv2.circle(mask, (x, y), 35, 0, -1)
        
        # Apply mask to get edges only
        edges_only = cv2.bitwise_and(gray, mask)
        
        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            edges_only,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=30,
            maxLineGap=50
        )
        
        # For each detected line, find which nodes it connects
        node_pairs = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Find the two nodes this line connects
                node1_idx = self._find_closest_node(nodes, (x1, y1))
                node2_idx = self._find_closest_node(nodes, (x2, y2))
                
                if node1_idx != node2_idx and node1_idx is not None and node2_idx is not None:
                    # Calculate midpoint for weight detection
                    mid_x = (x1 + x2) // 2
                    mid_y = (y1 + y2) // 2
                    
                    # Extract weight from the midpoint area
                    weight = self._extract_weight_at_position(img, mid_x, mid_y)
                    
                    if weight is not None:
                        edges.append((node1_idx, node2_idx, weight))
        
        # Remove duplicate edges
        unique_edges = []
        seen = set()
        for n1, n2, w in edges:
            edge_key = tuple(sorted([n1, n2]))
            if edge_key not in seen:
                seen.add(edge_key)
                unique_edges.append((n1, n2, w))
        
        return unique_edges
    
    def _find_closest_node(self, nodes: List[Tuple[int, int]], point: Tuple[int, int], threshold: int = 60) -> int:
        """Find the closest node to a given point"""
        min_dist = float('inf')
        closest_idx = None
        
        for idx, node in enumerate(nodes):
            dist = np.sqrt((node[0] - point[0])**2 + (node[1] - point[1])**2)
            if dist < min_dist and dist < threshold:
                min_dist = dist
                closest_idx = idx
        
        return closest_idx
    
    def _extract_weight_at_position(self, img: np.ndarray, x: int, y: int, region_size: int = 30) -> int:
        """Extract the weight number from a region around the given position"""
        # Define region around the midpoint
        y1 = max(0, y - region_size)
        y2 = min(img.shape[0], y + region_size)
        x1 = max(0, x - region_size)
        x2 = min(img.shape[1], x + region_size)
        
        # Extract region
        region = img[y1:y2, x1:x2]
        
        # Try OCR on the region
        try:
            # Preprocess for better OCR
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray_region, 127, 255, cv2.THRESH_BINARY)
            
            # Use pytesseract to extract text
            text = pytesseract.image_to_string(
                binary,
                config='--psm 8 -c tessedit_char_whitelist=0123456789'
            ).strip()
            
            if text.isdigit():
                return int(text)
        except Exception as e:
            logging.warning(f"OCR failed at position ({x}, {y}): {e}")
        
        return None
    
    def extract_graph(self, img: np.ndarray) -> Tuple[int, List[Tuple[int, int, int]]]:
        """Extract the complete graph structure from the image"""
        nodes = self.detect_nodes(img)
        edges = self.detect_edges_and_weights(img, nodes)
        
        # If edge detection with OCR fails, try alternative method
        if not edges or any(e[2] is None for e in edges):
            edges = self._extract_edges_alternative(img, nodes)
        
        return len(nodes), edges
    
    def _extract_edges_alternative(self, img: np.ndarray, nodes: List[Tuple[int, int]]) -> List[Tuple[int, int, int]]:
        """Alternative method to extract edges using color detection"""
        edges = []
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Try to detect colored edges and their weights
        # This is a simplified approach - you may need to adjust based on actual images
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                # Check if there's a line between nodes i and j
                x1, y1 = nodes[i]
                x2, y2 = nodes[j]
                
                # Sample points along the potential edge
                num_samples = 10
                edge_exists = True
                
                for k in range(1, num_samples):
                    t = k / num_samples
                    sample_x = int(x1 + t * (x2 - x1))
                    sample_y = int(y1 + t * (y2 - y1))
                    
                    # Check if there's a non-white pixel at this position
                    pixel = img[sample_y, sample_x]
                    if np.all(pixel > 240):  # If pixel is white
                        edge_exists = False
                        break
                
                if edge_exists:
                    # Extract weight from midpoint
                    mid_x = (x1 + x2) // 2
                    mid_y = (y1 + y2) // 2
                    
                    # Try to extract weight (simplified - may need adjustment)
                    weight = self._extract_weight_simple(img, mid_x, mid_y)
                    if weight is not None:
                        edges.append((i, j, weight))
        
        return edges
    
    def _extract_weight_simple(self, img: np.ndarray, x: int, y: int) -> int:
        """Simplified weight extraction"""
        # This is a placeholder - in practice, you'd need more sophisticated OCR
        # or pattern matching based on the actual image format
        region_size = 20
        y1 = max(0, y - region_size)
        y2 = min(img.shape[0], y + region_size)
        x1 = max(0, x - region_size)
        x2 = min(img.shape[1], x + region_size)
        
        region = img[y1:y2, x1:x2]
        
        # Try basic OCR
        try:
            text = pytesseract.image_to_string(region, config='--psm 8 digits').strip()
            if text.isdigit():
                return int(text)
        except:
            pass
        
        # Return a default weight if extraction fails (this should be improved)
        return 1


class UnionFind:
    """Union-Find data structure for Kruskal's algorithm"""
    
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> bool:
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True


def kruskal_mst(num_nodes: int, edges: List[Tuple[int, int, int]]) -> int:
    """Calculate MST weight using Kruskal's algorithm"""
    if not edges:
        return 0
    
    # Sort edges by weight
    edges.sort(key=lambda x: x[2])
    
    uf = UnionFind(num_nodes)
    mst_weight = 0
    edges_added = 0
    
    for u, v, weight in edges:
        if uf.union(u, v):
            mst_weight += weight
            edges_added += 1
            if edges_added == num_nodes - 1:
                break
    
    return mst_weight


def process_image(base64_image: str) -> int:
    """Process a single image and return MST weight"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to OpenCV format
        img_array = np.array(image)
        if img_array.shape[2] == 4:  # RGBA to RGB
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Extract graph structure
        extractor = GraphExtractor()
        num_nodes, edges = extractor.extract_graph(img_array)
        
        # Calculate MST
        mst_weight = kruskal_mst(num_nodes, edges)
        
        return mst_weight
    
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        raise


@app.route('/mst-calculation', methods=['POST'])
def mst_calculation():
    """API endpoint for MST calculation"""
    try:
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({"error": "Input must be a list"}), 400
        
        results = []
        
        for test_case in data:
            if 'image' not in test_case:
                return jsonify({"error": "Each test case must have an 'image' field"}), 400
            
            mst_weight = process_image(test_case['image'])
            results.append({"value": mst_weight})
        
        return jsonify(results)
    
    except Exception as e:
        logging.error(f"Error in mst_calculation endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)