import json
import logging
import math
from collections import defaultdict

from flask import request, jsonify

from routes import app

logger = logging.getLogger(__name__)

def calculate_distance(point1, point2):
    dx = point1[0] - point2[0]
    dy = point1[1] - point2[1]
    return math.sqrt(dx * dx + dy * dy)

def calculate_points(customer, concert, priority):
    points = 0

    # Factor 1: VIP Status
    if customer.get('vip_status', False):
        points += 100
    
    # Factor 2: Credit Card 
    if priority.get(customer.get('credit_card')) == concert.get('name'):
        points += 50
    
    # Factor 3: Latency
    distance = calculate_distance(
        customer.get('location', [0, 0]),
        concert.get('booking_center_location', [0, 0])
    )
    return (points, distance)

@app.route('/ticketing-agent', methods=['POST'])
def ticketing_agent():
    try:
        data = request.get_json()
        customers = data.get('customers', [])
        concerts = data.get('concerts', [])
        priority = data.get('priority', {})
        customer_odds = defaultdict(dict)
        res = dict()

        for concert in concerts:
            name = concert.get('name')
            customer_dist = dict()
            customer_points = dict()

            for customer in customers:
                c_name = customer.get('name')
                points, distance = calculate_points(customer, concert, priority)
                customer_dist[c_name] = distance
                customer_points[c_name] = points
            max_dist = max(customer_dist.values())
            min_dist = min(customer_dist.values())
            for customer in customers:
                c_name = customer.get('name')
                dist = customer_dist[c_name]
                if max_dist - min_dist == 0:
                    latency_score = 30  # Give everyone max latency points
                    customer_points[c_name] += latency_score
                else:
                    latency_score = dist / (max_dist - min_dist) * 30
                    customer_points[c_name] += (30 - latency_score)
            total_points = sum(customer_points.values())
            if total_points > 0:
                for customer in customers:
                    c_name = customer.get('name')
                    customer_points[c_name] = (customer_points[c_name] / total_points) * 100
                    customer_odds[c_name][name] = customer_points[c_name]
            else:
                # If total points is 0, give equal odds to all customers
                for customer in customers:
                    c_name = customer.get('name')
                    customer_odds[c_name][name] = 100 / len(customers)
        for customer in customer_odds:
            max_odds = max(customer_odds[customer].values())
            for concert in customer_odds[customer]:
                if customer_odds[customer][concert] == max_odds:
                    res[customer] = concert
                    break
        return jsonify(res)
    
    except Exception as e:
        logger.error(f'Error processing request: {str(e)}')
        return jsonify({'error': 'Internal server error'}), 500
