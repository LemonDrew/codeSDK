from flask import Flask, jsonify

from routes import app

@app.route('/trivia', methods=['GET'])
def get_trivia_answers():
    answers = [
        2,  # Q1: Educated guess - likely 3 challenges
        1,  # Q2: Concert tickets (most common)
        3,  # Q3: 1000 lists x 100 elements (reasonable dataset size)
        2,  # Q4: Fat Louie (correct answer from Princess Diaries)
        4,  # Q5: 10 nodes (reasonable average for MST problems)
        4,  # Q6: Amy Winehouse (never did a James Bond theme)
        2,  # Q7: 0.5px (smallest reasonable font size)
        1,  # Q8: "graceful the pet" (anagram of "capture the flag")
        2   # Q9: Australia, Hong Kong, Japan, Singapore (comprehensive list)
    ]
    
    return jsonify({
        "answers": answers
    })

if __name__ == '__main__':
    app.run(debug=True)