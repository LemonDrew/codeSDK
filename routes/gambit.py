from flask import Flask, request, jsonify
from routes import app

@app.route("/the-mages-gambit", methods=["POST"])
def gambit():
    data = request.get_json()
    result = []

    for entry in data:

        intel = entry["intel"]
        reserve = entry["reserve"]
        stamina = entry["stamina"]

        current_mp = reserve
        current_stamina = stamina
        total = 0
        prev_front = None

        for front in intel:
            number, mana_needed = front

            if prev_front is not None and prev_front != number:
                current_stamina = stamina  

            if current_mp < mana_needed or current_stamina == 0:
                current_mp = reserve
                current_stamina = stamina
                total += 10  

            current_mp -= mana_needed
            current_stamina -= 1
            total += 10  

            prev_front = number
        
        result.append({"time" : total + 10})


    return jsonify(result)
   

if __name__ == "__main__":
    app.run(port=3000, debug=True)