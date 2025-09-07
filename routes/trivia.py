from flask import jsonify

from routes import app

@app.route('/trivia', methods=['GET'])
def get_trivia_answers():
    answers = [
        3,  # 1
        1,  # 2
        2,  # 3
        2,  # 4
        3,  # 5
        4,  # 6
        3,  # 7: ?
        1,  # 8
        3,  # 9
        3, #10
        3, #11
        3, #12
        4, #13
        1, #14: ?
        2, #15
        1, #16
        1, #17
        2, #18
        2, #19: ?
        1, #20
        1, #21: ?
        2, #22
        3, #23
        5, #24
        2 #25
    ]
    
    return jsonify({
        "answers": answers
    })