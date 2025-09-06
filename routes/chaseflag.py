from flask import Flask, jsonify, request

from routes import app

@app.route('/chasetheflag', methods=['POST'])
def chaseflag():
    """
    Handle POST requests to /chasetheflag endpoint
    Returns JSON response with challenge flags
    """
    response_data = {
        "challenge1": "O0qEMh7+oA1ckgB5O2uwzyYyhiA=",
        "challenge2": "",
        "challenge3": "98562706cef8",
        "challenge4": "",
        "challenge5": ""
    }
    
    return jsonify(response_data)