from flask import Flask, jsonify

from routes import app

@app.route('/trivia', methods=['GET'])
def get_trivia_answers():
    answers = [
        3,  # Q1: Educated guess 
        1,  # Q2
        2,  # Q3
        2,  # Q4
        3,  # Q5
        4,  # Q6: Amy Winehouse (never did a James Bond theme)
        2,  # Q7: 0.5px (smallest reasonable font size)
        1,  # Q8: "graceful the pet" (anagram of "capture the flag")
        3   # Q9: USA, Hong Kong, Singapore
    ]
    
    return jsonify({
        "answers": answers
    })

if __name__ == '__main__':
    app.run(debug=True)