from flask import Flask, jsonify

from routes import app

@app.route('/trivia', methods=['GET'])
def get_trivia_answers():
    answers = [
        3,  # Q1: ?
        1,  # Q2
        2,  # Q3
        2,  # Q4
        3,  # Q5
        4,  # Q6
        1,  # Q7: ?
        1,  # Q8
        3,  # Q9
        3, #10
        3, #11
        3, #12
        4, #13
        1, #14: ?
        2, #15
        1, #16
        1, #17
        2, #18
        1, #19: ?
        1, #20
        1, #21: ?
        2, #22
        2, #23: ?
        5, #24
        2 #25
    ]
    
    return jsonify({
        "answers": answers
    })

if __name__ == '__main__':
    app.run(debug=True)