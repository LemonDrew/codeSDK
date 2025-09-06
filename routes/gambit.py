from flask import Flask, request, jsonify
from routes import app

@app.route("/the-mages-gambit", methods=["POST"])
def gambit():
    data = request.get_json()
    result = []

    for entry in data:
        intel = entry["intel"]
        reserve = entry["reserve"]
        fronts = entry["fronts"]
        stamina  = entry["stamina"]

        current_mp = reserve
        current_stamina = stamina

        total = 0
        prev_front = 0

        for front in intel:
            
            number, amount = front

            if prev_front == 0:
                current_stamina -= 1
                current_mp -= amount
                total += 10

                prev_front = number

                continue

            if prev_front == number:
                if current_stamina >=1 and current_mp >= amount:
                    current_stamina -= 1
                    current_mp -= amount
                    continue

                else:
                    current_stamina = stamina
                    current_mp = reserve
                    total += 10
                    continue

            else: 
                if current_stamina >=1 and current_mp >= amount:
                    current_stamina -= 1
                    current_mp -= amount
                    
    
                else:
                    current_stamina = stamina
                    current_mp = reserve        
                    total += 10

                    current_stamina -= 1
                    current_mp -= amount
                
                total += 10

        result.append({"time" : total + 10})

    return result



if __name__ == "__main__":
    app.run(port=3000, debug=True)