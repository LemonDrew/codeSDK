from flask import Flask, jsonify, request

from routes import app

@app.route('/chasetheflag', methods=['POST'])
def chaseflag():
    response_data = {
        "challenge1": "UBS479af657c4de",
        "challenge2": "",
        "challenge3": "UBS98562706cef8",
        "challenge4": "",
        "challenge5": ""
    }
    
    return jsonify(response_data)