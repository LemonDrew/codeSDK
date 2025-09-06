from flask import Flask, request, jsonify
import re # For regular expressions, useful for tokenizing and cleaning

from routes import app

# --- Helper Functions for Number Conversion ---

# Roman Numeral to Integer mapping
ROMAN_MAP = {
    'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000
}

def roman_to_int(s: str) -> int:
    """Converts a Roman numeral string to an integer."""
    result = 0
    prev_val = 0
    for char in reversed(s):
        curr_val = ROMAN_MAP[char]
        if curr_val < prev_val:
            result -= curr_val
        else:
            result += curr_val
        prev_val = curr_val
    return result

# English words to Integer mapping (expanded for more coverage)
ENGLISH_MAP = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90
}
ENGLISH_MAGNITUDES = {
    "hundred": 100, "thousand": 1000, "million": 1000000, "billion": 1000000000
}

def english_to_int(s: str) -> int:
    """
    Converts an English number string to an integer.
    Handles compound numbers and magnitudes.
    """
    s = s.replace('-', ' ').lower()
    words = [word for word in s.split() if word not in ["and"]] # "and" is often ignored in parsing
    
    total = 0
    current_number = 0
    
    for word in words:
        if word in ENGLISH_MAP:
            current_number += ENGLISH_MAP[word]
        elif word in ENGLISH_MAGNITUDES:
            magnitude_val = ENGLISH_MAGNITUDES[word]
            if magnitude_val > 100: # For thousands, millions, etc.
                total += current_number * magnitude_val
                current_number = 0
            else: # For hundreds
                current_number *= magnitude_val
        else:
            # Handle cases like "one hundred twenty" where "twenty" adds to current_number
            # If `current_number` holds a multiple of 100 (e.g., from "one hundred"),
            # then subsequent words like "twenty" should add to it.
            # This simplified logic might need further refinement for complex phrases.
            pass
            
    return total + current_number

# German words to Integer mapping (expanded and refined)
GERMAN_MAP_UNITS = {
    "null": 0, "ein": 1, "eins": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5,
    "sechs": 6, "sieben": 7, "acht": 8, "neun": 9
}
GERMAN_MAP_TEENS = {
    "zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14,
    "fünfzehn": 15, "sechzehn": 16, "siebzehn": 17, "achtzehn": 18,
    "neunzehn": 19
}
GERMAN_MAP_TENS = {
    "zwanzig": 20, "dreißig": 30, "vierzig": 40, "fünfzig": 50,
    "sechzig": 60, "siebzig": 70, "achtzig": 80, "neunzig": 90
}
GERMAN_MAP_MAGNITUDES = {
    "hundert": 100, "tausend": 1000, "million": 1000000, "milliarde": 1000000000
}

def german_to_int(s: str) -> int:
    """
    Converts a German number string to an integer.
    Handles concatenated words (e.g., "siebenundachtzig") and magnitudes.
    """
    s = s.lower().replace("ß", "ss").replace("-", "") # Normalize
    
    total = 0
    current_segment = 0 # Holds the value of the current "block" (e.g., up to 999)
    last_magnitude = 1 # To handle e.g. "zwei millionen fünf hundert tausend"

    # Pre-parse concatenated numbers like "siebenundachtzig"
    # This is a common pattern: units-und-tens (e.g., sieben-und-achtzig = 7 + 80 = 87)
    # This loop needs to be careful not to over-split for numbers like "einhundert"
    
    # Simple direct lookups for common single words that are exceptions or key examples
    if s in GERMAN_MAP_UNITS: return GERMAN_MAP_UNITS[s]
    if s in GERMAN_MAP_TEENS: return GERMAN_MAP_TEENS[s]
    if s in GERMAN_MAP_TENS: return GERMAN_MAP_TENS[s]
    
    # Handle numbers that are often fully concatenated (like "dreihundertelf")
    # This requires a more programmatic approach than just splitting by spaces/und.
    # For now, let's try to parse parts.
    
    # First, handle 'und' for units-and-tens (e.g., "siebenundachtzig")
    if 'und' in s:
        parts = s.split('und')
        if len(parts) == 2:
            unit_part = parts[0]
            ten_part = parts[1]
            unit_val = GERMAN_MAP_UNITS.get(unit_part, 0)
            
            # Find the tens value from the ten_part string
            ten_val = 0
            for k, v in GERMAN_MAP_TENS.items():
                if ten_part.endswith(k) and len(ten_part) == len(k): # Exact match for ten part like "achtzig"
                    ten_val = v
                    break
            
            if unit_val > 0 and ten_val > 0:
                return ten_val + unit_val # e.g., 80 + 7 = 87
    
    # Now, parse numbers that can be constructed sequentially or with magnitudes
    # Split the string into potential number words.
    words = re.findall(r'[a-zäöüß]+', s) # Get all alphabetic sequences
    
    for word in words:
        if word in GERMAN_MAP_UNITS:
            current_segment += GERMAN_MAP_UNITS[word]
        elif word in GERMAN_MAP_TEENS:
            current_segment += GERMAN_MAP_TEENS[word]
        elif word in GERMAN_MAP_TENS:
            current_segment += GERMAN_MAP_TENS[word]
        elif word in GERMAN_MAP_MAGNITUDES:
            magnitude_val = GERMAN_MAP_MAGNITUDES[word]
            if magnitude_val == 100: # Hundert
                if current_segment == 0: current_segment = 1 # "hundert" implies "ein hundert"
                current_segment *= magnitude_val
            else: # Tausend, Millionen, etc.
                total += current_segment * magnitude_val
                current_segment = 0
                last_magnitude = magnitude_val # Update last magnitude
        elif word == "ein" and current_segment == 0: # special case for "ein hundert", "ein tausend"
             current_segment = 1
        
    return total + current_segment # Add the final segment

# Chinese characters to Integer mapping and conversion logic (refined)
CHINESE_NUMERALS = {
    '零': 0, '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
    '六': 6, '七': 7, '八': 8, '九': 9, '十': 10, '百': 100,
    '千': 1000, '萬': 10000, '亿': 100000000, # Simplified + common '亿'
    '万': 10000, '億': 100000000 # Traditional + common '萬'
}

def chinese_to_int(s: str) -> int:
    """
    Converts a Chinese number string (Traditional or Simplified) to an integer.
    Handles magnitudes and standard Chinese number structures.
    """
    result = 0
    current_chunk = 0 # Value of the current 'segment' (e.g., 4321 in 54321)
    # The current_chunk handles numbers up to '千' (1000) within a '萬' or '億' block.
    
    # Magnitude to multiply current_chunk by before adding to result
    # For example, if we see '萬', current_chunk is multiplied by 10000
    
    temp_num = 0 # Stores the value of individual digits or small multiples (e.g., '四' in '四千')
    
    # This logic needs to be stateful. Process characters one by one.
    
    # High-level multipliers (万/萬, 亿/億)
    high_multiplier = 1
    
    for char in s:
        if char in CHINESE_NUMERALS:
            val = CHINESE_NUMERALS[char]
            if 0 <= val <= 9: # Digits
                temp_num = val
            elif val == 10: # 十
                if temp_num == 0: # Cases like "十" (10)
                    temp_num = 1
                current_chunk += temp_num * val
                temp_num = 0
            elif val == 100: # 百
                if temp_num == 0: temp_num = 1 # Cases like "百" (implied "一百")
                current_chunk += temp_num * val
                temp_num = 0
            elif val == 1000: # 千
                if temp_num == 0: temp_num = 1 # Cases like "千" (implied "一千")
                current_chunk += temp_num * val
                temp_num = 0
            elif val == 10000 or val == 100000000: # 萬/万, 億/亿
                # Add the accumulated chunk to the result, multiplied by the current high_multiplier
                result += (current_chunk + temp_num) * val
                current_chunk = 0
                temp_num = 0
                high_multiplier = val # Update the high multiplier (not directly used in this logic, but conceptually)
        else:
            # Unrecognized character, should not happen with valid input.
            pass
            
    # Add any remaining value in current_chunk or temp_num
    result += current_chunk + temp_num
    return result

# --- Language Type Detection and Value Mapping ---

# Order of representation for tie-breaking in Part 2
REPRESENTATION_ORDER = {
    "roman": 0,
    "english": 1,
    "traditional_chinese": 2,
    "simplified_chinese": 3,
    "german": 4,
    "arabic": 5
}

def get_number_info(num_str: str):
    """
    Determines the type and integer value of a number string.
    Returns (integer_value, language_type_string).
    """
    original_num_str = num_str # Keep original for potential errors/logging
    num_str = num_str.strip()

    # 1. Arabic Numerals
    if num_str.isdigit() or (num_str.startswith('-') and num_str[1:].isdigit()):
        return int(num_str), "arabic"

    # 2. Roman Numerals
    # Check if all chars are Roman and it's not empty, and ensure it's not a mixed string
    # A more robust check might involve regex for valid Roman numeral patterns.
    is_roman = True
    if num_str and all(c in ROMAN_MAP for c in num_str.upper()):
        for char in num_str:
            if not char.isalpha(): # Ensure no numbers or other symbols
                is_roman = False
                break
        if is_roman:
            try:
                return roman_to_int(num_str.upper()), "roman"
            except:
                pass # Conversion failed, not a valid Roman numeral

    # 3. Chinese (Traditional/Simplified)
    contains_chinese_chars = any(c in CHINESE_NUMERALS for c in num_str)
    if contains_chinese_chars:
        # Check for specific characters to differentiate for tie-breaking
        has_traditional_wan = '萬' in num_str
        has_simplified_wan = '万' in num_str

        try:
            val = chinese_to_int(num_str)
            if has_traditional_wan and not has_simplified_wan:
                return val, "traditional_chinese"
            elif has_simplified_wan and not has_traditional_wan:
                return val, "simplified_chinese"
            elif has_traditional_wan and has_simplified_wan: # Ambiguous, prioritize traditional if both are present
                return val, "traditional_chinese"
            else: # No specific '万' or '萬', but definitely Chinese characters
                # Default to simplified if ambiguous, or use another heuristic
                # For this problem, let's assume if it's Chinese characters, one of these will be sufficient.
                # If numbers like '一二三' are given without units, they are ambiguous.
                # Assuming the challenge inputs will provide enough context.
                return val, "simplified_chinese" # Default if no specific wan/wan
        except Exception as e:
            # print(f"Chinese conversion failed for '{num_str}': {e}")
            pass

    # 4. English Words
    # Stronger check: look for common English number words and avoid German/Roman
    # Use a regex to look for typical English number word patterns, separated by spaces/hyphens
    english_keywords = set(ENGLISH_MAP.keys()).union(set(ENGLISH_MAGNITUDES.keys()))
    words = re.findall(r'[a-z]+', num_str.lower())
    
    is_likely_english = False
    if len(words) > 0 and any(word in english_keywords for word in words):
        # Basic check to avoid misclassifying German or Roman
        if not any(c in 'äöüß' for c in num_str.lower()) and not any(c in ROMAN_MAP for c in num_str.upper()):
            is_likely_english = True
            
    if is_likely_english:
        try:
            return english_to_int(num_str), "english"
        except Exception as e:
            # print(f"English conversion failed for '{num_str}': {e}")
            pass

    # 5. German Words
    # Check for strong German indicators (e.g., "und", "hundert", umlauts, eszett)
    # Be careful not to misclassify "one hundred" as German "ein hundert"
    german_keywords = set(GERMAN_MAP_UNITS.keys()).union(set(GERMAN_MAP_TEENS.keys())).union(
                        set(GERMAN_MAP_TENS.keys())).union(set(GERMAN_MAP_MAGNITUDES.keys()))
    words_german = re.findall(r'[a-zäöüß]+', num_str.lower().replace("ß", "ss"))
    
    is_likely_german = False
    if len(words_german) > 0 and any(word in german_keywords for word in words_german):
        # More specific German checks: "und" in combined words, umlauts, eszett
        if 'und' in num_str.lower() or any(c in 'äöüß' for c in num_str.lower()):
            is_likely_german = True
            
    if is_likely_german:
        try:
            return german_to_int(num_str), "german"
        except Exception as e:
            # print(f"German conversion failed for '{num_str}': {e}")
            pass

    # Fallback for unexpected cases (should ideally not be reached if inputs conform)
    raise ValueError(f"Could not parse number: {original_num_str}")


# --- Flask Endpoint ---

@app.route('/duolingo-sort', methods=['POST'])
def duolingo_sort():
    data = request.json
    part = data['part']
    unsorted_list = data['challengeInput']['unsortedList']

    try:
        if part == "ONE":
            processed_list = []
            for item in unsorted_list:
                val, _ = get_number_info(item)
                processed_list.append(val)
            
            processed_list.sort()
            sorted_list_str = [str(x) for x in processed_list]
            
            return jsonify({"sortedList": sorted_list_str})

        elif part == "TWO":
            items_to_sort = []
            for item_str in unsorted_list:
                int_val, lang_type = get_number_info(item_str)
                rep_order = REPRESENTATION_ORDER.get(lang_type, 999) # Default to high priority for safety
                items_to_sort.append((int_val, rep_order, item_str))
            
            items_to_sort.sort() # Sorts by int_val then rep_order
            
            sorted_list_original_format = [item[2] for item in items_to_sort]
            
            return jsonify({"sortedList": sorted_list_original_format})

        else:
            return jsonify({"error": "Invalid part specified. Use 'ONE' or 'TWO'."}), 400
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
