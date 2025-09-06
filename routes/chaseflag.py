from flask import Flask, jsonify, request

from routes import app

@app.route('/chasetheflag', methods=['POST'])
def chaseflag():
    response_data = {
        "challenge1": "UBS{479af657c4de}",
        "challenge2": "",
        "challenge3": "UBS{98562706cef8}",
        "challenge4": "",
        "challenge5": "UBS{98cb49b7e5fcd72d63769b53ec41639c}"
    }
    
    return jsonify(response_data)