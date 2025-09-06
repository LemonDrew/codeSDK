import re
from flask import Flask, request, jsonify
from routes import app

def process_challenge_one(data):

    transform_map = {
        "encode_mirror_alphabet": encode_mirror_alphabet,
        "double_consonants": double_consonants,
        "mirror_words": mirror_words,
        "swap_pairs": swap_pairs,
        "encode_index_parity": encode_index_parity
    }

    transformations = data["transformations"]
    encrypted_data = data["transformed_encrypted_word"]

    func_names = re.findall(r'(\w+)\(x\)', transformations)

    for function_name in func_names:
        func = transform_map[function_name]
        encrypted_data = func(encrypted_data)

    return encrypted_data

def mirror_words(x):

    words = x.split(" ")
    result = []

    for word in words:
        reversed_word = word[::-1]
        result.append(reversed_word)

    print(result)

    return (" ").join(result)


def encode_mirror_alphabet(x):
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
    swapped_words = []

    for word in words:
        chars = list(word)
        for i in range(0, len(chars) - 1, 2):
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        swapped_words.append("".join(chars))

    return " ".join(swapped_words)


def encode_index_parity(x):
    words = x.split(" ")
    transformed_words = []

    for word in words:
        even_chars = [word[i] for i in range(0, len(word), 2)]
        odd_chars = [word[i] for i in range(1, len(word), 2)]
        transformed_words.append("".join(even_chars + odd_chars))

    return " ".join(transformed_words)

def double_consonants(x):
    vowels = set("aeiouAEIOU")
    result = []

    for ch in x:
        if ch.isalpha() and ch not in vowels:  # Consonant
            result.append(ch * 2)
        else:  # Vowel or non-letter
            result.append(ch)

    return "".join(result)


@app.route("/operation-safeguard", methods=["POST"])
def operation_safeguard():
    
    data = request.get_json()

    challenge_one = data["challenge_one"]

    result_one = process_challenge_one(challenge_one)


    return jsonify({
        "challenge_one": result_one,
        "challenge_two": "value_from_challenge_2",
        "challenge_three": "value_from_challenge_3",
        "challenge_four": "final_decrypted_value"
    })

if __name__ == "__main__":
    app.run(port=3000, debug=True)

