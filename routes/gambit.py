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
        total_time = 0
        prev_front = None

        for number, amount in intel:
            if prev_front != number:
                total_time += 10
                prev_front = number

            remaining_mp_needed = amount

            while remaining_mp_needed > 0:
                if current_mp == 0 or current_stamina == 0:
                    total_time += 10
                    current_mp = reserve
                    current_stamina = stamina

                used_mp = min(current_mp, remaining_mp_needed)
                current_mp -= used_mp
                current_stamina -= 1
                remaining_mp_needed -= used_mp

                if remaining_mp_needed > 0 and (current_mp == 0 or current_stamina == 0):
                    total_time += 10
                    current_mp = reserve
                    current_stamina = stamina

        total_time += 10

        result.append({"time": total_time})

    return jsonify(result)


if __name__ == "__main__":
    app.run(port=3000, debug=True)