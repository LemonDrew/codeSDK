from flask import Flask, request, jsonify
import re

from routes import app

# =======================
# Roman numeral parser
# =======================
roman_map = {
    'I': 1, 'V': 5, 'X': 10, 'L': 50,
    'C': 100, 'D': 500, 'M': 1000
}

def roman_to_int(s):
    total = 0
    prev = 0
    for c in reversed(s):
        val = roman_map[c]
        if val < prev:
            total -= val
        else:
            total += val
        prev = val
    return total

# =======================
# English parser
# =======================
eng_nums = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
}
eng_scales = {
    "hundred": 100,
    "thousand": 1000,
    "million": 1000000,
    "billion": 1000000000
}

def english_to_int(text):
    words = text.replace("-", " ").split()
    total = 0
    current = 0
    for w in words:
        if w in eng_nums:
            current += eng_nums[w]
        elif w in eng_scales:
            current *= eng_scales[w]
            if eng_scales[w] >= 1000:
                total += current
                current = 0
        elif w == "and":
            continue
        else:
            raise ValueError(f"Unknown English number word: {w}")
    return total + current

# =======================
# German parser
# =======================
german_units = {
    "null": 0, "eins": 1, "ein": 1, "zwei": 2, "drei": 3, "vier": 4,
    "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8, "neun": 9,
    "zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13,
    "vierzehn": 14, "fünfzehn": 15, "sechzehn": 16,
    "siebzehn": 17, "achtzehn": 18, "neunzehn": 19
}
german_tens = {
    "zwanzig": 20, "dreißig": 30, "vierzig": 40,
    "fünfzig": 50, "sechzig": 60, "siebzig": 70,
    "achtzig": 80, "neunzig": 90
}
german_scales = {
    "hundert": 100, "tausend": 1000
}

def german_to_int(text):
    text = text.lower()
    if text in german_units:
        return german_units[text]
    if text in german_tens:
        return german_tens[text]
    # handle "siebenundachtzig"
    m = re.match(r"(.+)und(.+)", text)
    if m:
        left, right = m.groups()
        left_val = german_units.get(left, 0)
        right_val = german_tens.get(right, 0)
        if right_val:
            return left_val + right_val
    # handle hundreds/thousands like "dreihundertelf"
    for scale_word, scale_val in german_scales.items():
        if scale_word in text:
            parts = text.split(scale_word, 1)
            prefix = parts[0]
            suffix = parts[1] if len(parts) > 1 else ""
            total = 0
            if prefix:
                total += german_units.get(prefix, 1) * scale_val
            else:
                total += scale_val
            if suffix:
                total += german_to_int(suffix)
            return total
    raise ValueError(f"Unknown German number: {text}")

# =======================
# Chinese parser
# =======================
chinese_digits = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9,
    "壹": 1, "貳": 2, "參": 3, "肆": 4, "伍": 5, "陸": 6, "柒": 7,
    "捌": 8, "玖": 9,
}
chinese_units = {
    "十": 10, "百": 100, "千": 1000,
    "萬": 10000, "万": 10000,
    "億": 100000000, "亿": 100000000
}

def chinese_to_int(s):
    total = 0
    section = 0
    number = 0
    for ch in s:
        if ch in chinese_digits:
            number = chinese_digits[ch]
        elif ch in chinese_units:
            unit = chinese_units[ch]
            if unit >= 10000:
                section = (section + number) * unit
                total += section
                section = 0
            else:
                if number == 0:
                    number = 1
                section += number * unit
            number = 0
        else:
            raise ValueError(f"Unknown Chinese char: {ch}")
    return total + section + number

# =======================
# Language detection
# =======================
traditional_chars = set("萬億壹貳參肆伍陸柒捌玖")
simplified_chars = set("万亿两〇")

def detect_language(s):
    if re.fullmatch(r"[IVXLCDM]+", s):
        return "roman"
    if s.isdigit():
        return "arabic"
    if re.search(r"[零一二三四五六七八九十百千万萬億亿壹貳參肆伍陸柒捌玖两〇]", s):
        if any(ch in s for ch in traditional_chars):
            return "traditional"
        if any(ch in s for ch in simplified_chars):
            return "simplified"
        # default ambiguous numbers like "四十五"
        return "traditional"
    if any(w in s.lower() for w in eng_nums.keys() | eng_scales.keys() | {"and"}):
        return "english"
    return "german"

def to_int(s):
    lang = detect_language(s)
    if lang == "roman":
        return roman_to_int(s)
    if lang == "arabic":
        return int(s)
    if lang == "english":
        return english_to_int(s.lower())
    if lang == "german":
        return german_to_int(s.lower())
    if lang in ("traditional", "simplified"):
        return chinese_to_int(s)
    raise ValueError(f"Could not detect language for {s}")

lang_order = {
    "roman": 0,
    "english": 1,
    "traditional": 2,
    "simplified": 3,
    "german": 4,
    "arabic": 5
}

# =======================
# Endpoint
# =======================
@app.route("/duolingo-sort", methods=["POST"])
def duolingo_sort():
    data = request.get_json()
    part = data.get("part")
    unsorted_list = data.get("challengeInput", {}).get("unsortedList", [])

    if part == "ONE":
        sorted_list = sorted(
            [str(to_int(x)) for x in unsorted_list],
            key=lambda x: int(x)
        )
    else:  # TWO
        sorted_list = sorted(
            unsorted_list,
            key=lambda s: (to_int(s), lang_order[detect_language(s)])
        )

    return jsonify({"sortedList": sorted_list})
