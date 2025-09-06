from flask import Flask, jsonify

from routes import app

@app.route('/trivia', methods=['GET'])
def get_trivia_answers():
    answers = [
        2,  # Q1: Educated guess 
        1,  # Q2
        2,  # Q3
        2,  # Q4
        3,  # Q5
        4,  # Q6: Amy Winehouse (never did a James Bond theme)
        3,  # Q7: 0.5px (smallest reasonable font size)
        1,  # Q8: "graceful the pet" (anagram of "capture the flag")
        3,  # Q9: USA, Hong Kong, Singapore
        3, #10
        3, #11
        3, #12
        4, #13,
        2,
        2, #15
        2,
        1,
        2, #18
        1,
        1, #20
        2,
        2, #22
        2,
        5,
        2 #25
    ]
    
    return jsonify({
        "answers": answers
    })

if __name__ == '__main__':
    app.run(debug=True)