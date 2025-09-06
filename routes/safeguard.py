from flask import Flask, request, jsonify
from routes import app
# import logging

# logger = logging.getLogger(__name__)


def mirror_words(x):

    words = x.split(" ")
    result = []

    for word in words:
        reversed_word = word[::-1]
        result.append(reversed_word)

    print(result)

    return (" ").join(result)


def decode_mirror_alphabet(x):
    result = []

    for ch in x:
        if 'a' <= ch <= 'z':
            mirrored = chr(ord('z') - (ord(ch) - ord('a')))
            result.append(mirrored)
        elif 'A' <= ch <= 'Z': 
            mirrored = chr(ord('Z') - (ord(ch) - ord('A')))
            result.append(mirrored)
        else:
            result.append(ch)

    print(result)

    return ''.join(result)


def toggle_case(x):

    result = []

    for ch in x:
        if ch.islower():
            result.append(ch.upper())   
        elif ch.isupper():
            result.append(ch.lower())    
        else:
            result.append(ch)            

    return ''.join(result)

def swap_pairs(x):
    words = x.split(" ")
    reversed_words = []

    for word in words:
        chars = list(word)
        last_char = None

    
        if len(chars) % 2 != 0:
            last_char = chars[-1]
            chars = chars[:-1]

        for i in range(0, len(chars), 2):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]

        if last_char:
            chars.append(last_char)

        reversed_words.append("".join(chars))

    return " ".join(reversed_words)



def decode_index_parity(x):
    words = x.split(" ")
    reversed_words = []

    for word in words:
        n = len(word)
        half = (n + 1) // 2  
        even_chars = word[:half]
        odd_chars = word[half:]

        original = []
        for i in range(n):
            if i % 2 == 0:
                original.append(even_chars[i // 2])
            else:
                original.append(odd_chars[i // 2])
        
        reversed_words.append("".join(original))

    return " ".join(reversed_words)


def remove_double_consonants(x):
    vowels = set("aeiouAEIOU")
    result = []
    i = 0

    while i < len(x):
        result.append(x[i])
        if x[i].isalpha() and x[i] not in vowels: 
            i += 2
        else:  
            i += 1

    return "".join(result)


def process_challenge_one(data):

    transform_map = {
        "encode_mirror_alphabet": decode_mirror_alphabet,
        "double_consonants": remove_double_consonants,
        "mirror_words": mirror_words,
        "swap_pairs": swap_pairs,
        "encode_index_parity": decode_index_parity
    }

    transformations = data["transformations"]

    if isinstance(transformations, str):
        transformations = transformations.strip("[]") 
        func_names = [name.strip() for name in transformations.split(",")]
    else:
        func_names = transformations

    new_transformations = []

    for f in func_names:

        order = []
        result = ""

        for char in f:
            if char == "(":
                order = [result] + order
                result = ""
                

            elif char == ")":
                new_transformations.extend(order)
                break

            else:
                result += char

    encrypted_data = data["transformed_encrypted_word"]

    for function_name in new_transformations:
        func = transform_map[function_name]
        encrypted_data = func(encrypted_data)

    return encrypted_data


@app.route("/operation-safeguard", methods=["POST"])
def operation_safeguard():
    
    data = request.get_json()

    challenge_one_data = data.get("challenge_one")
    challenge_two_data = data.get("challenge_two")
    challenge_three_data = data.get("challenge_three")
    challenge_four_data = data.get("challenge_four")

    result_one = process_challenge_one(challenge_one_data)



    if data is None:
        return jsonify({"error": "Invalid JSON"}), 400

    challenge_one_data = data.get("challenge_one")
    if challenge_one_data is None:
        return jsonify({"error": "Missing challenge_one"}), 400

    result_one = process_challenge_one(challenge_one_data)

    result_two = "value_from_challenge_2"
    result_three = "value_from_challenge_3"
    result_four = "final_decrypted_value"

    return jsonify({
        "challenge_one": result_one,
        "challenge_two": result_two,
        "challenge_three": result_three,
        "challenge_four": result_four
    })

if __name__ == "__main__":
    app.run(port=3000, debug=True)

