from flask import request, jsonify
import re
import json

from routes import app

class NumberParser:
    def __init__(self):
        # Roman numeral conversion
        self.roman_values = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000
        }
        
        # English number words
        self.english_ones = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19
        }
        
        self.english_tens = {
            'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50,
            'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
        }
        
        self.english_scales = {
            'hundred': 100, 'thousand': 1000, 'million': 1000000, 'billion': 1000000000
        }
        
        # German number words
        self.german_ones = {
            'null': 0, 'eins': 1, 'ein': 1, 'eine': 1, 'zwei': 2, 'drei': 3, 'vier': 4, 'fünf': 5,
            'sechs': 6, 'sieben': 7, 'acht': 8, 'neun': 9, 'zehn': 10,
            'elf': 11, 'zwölf': 12, 'dreizehn': 13, 'vierzehn': 14, 'fünfzehn': 15,
            'sechzehn': 16, 'siebzehn': 17, 'achtzehn': 18, 'neunzehn': 19
        }
        
        self.german_tens = {
            'zwanzig': 20, 'dreißig': 30, 'vierzig': 40, 'fünfzig': 50,
            'sechzig': 60, 'siebzig': 70, 'achtzig': 80, 'neunzig': 90
        }
        
        self.german_scales = {
            'hundert': 100, 'tausend': 1000, 'million': 1000000, 'milliarde': 1000000000
        }
        
        # Chinese numerals (both traditional and simplified)
        self.chinese_digits = {
            '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
            '〇': 0, '十': 10, '百': 100, '千': 1000, '萬': 10000, '万': 10000, '億': 100000000, '亿': 100000000
        }
    
    def roman_to_int(self, roman):
        """Convert Roman numeral to integer"""
        total = 0
        prev_value = 0
        
        for char in reversed(roman.upper()):
            value = self.roman_values.get(char, 0)
            if value < prev_value:
                total -= value
            else:
                total += value
            prev_value = value
        
        return total
    
    def english_to_int(self, text):
        """Convert English number words to integer"""
        text = text.lower().strip()
        
        # Handle simple cases
        if text in self.english_ones:
            return self.english_ones[text]
        
        total = 0
        current = 0
        
        words = text.split()
        i = 0
        
        while i < len(words):
            word = words[i]
            
            if word in self.english_ones:
                current += self.english_ones[word]
            elif word in self.english_tens:
                current += self.english_tens[word]
            elif word == 'hundred':
                current *= 100
            elif word in ['thousand', 'million', 'billion']:
                scale = self.english_scales[word]
                total += current * scale
                current = 0
            
            i += 1
        
        return total + current
    
    def german_to_int(self, text):
        """Convert German number words to integer"""
        text = text.lower().strip()
        
        # Handle compound numbers like "siebenundachtzig" (87)
        if 'und' in text:
            parts = text.split('und')
            if len(parts) == 2:
                ones_part = parts[0]
                tens_part = parts[1]
                
                ones_val = self.german_ones.get(ones_part, 0)
                tens_val = self.german_tens.get(tens_part, 0)
                
                if ones_val > 0 and tens_val > 0:
                    return tens_val + ones_val
        
        # Handle simple cases
        if text in self.german_ones:
            return self.german_ones[text]
        if text in self.german_tens:
            return self.german_tens[text]
        
        # Handle compound words like "dreihundertelf" (311)
        total = 0
        remaining = text
        
        # Check for scales in descending order
        for scale_word, scale_val in sorted(self.german_scales.items(), key=lambda x: x[1], reverse=True):
            if scale_word in remaining:
                parts = remaining.split(scale_word, 1)
                prefix = parts[0]
                suffix = parts[1] if len(parts) > 1 else ""
                
                multiplier = 1
                if prefix:
                    multiplier = self.german_ones.get(prefix, 0)
                    if multiplier == 0 and prefix in self.german_tens:
                        multiplier = self.german_tens[prefix]
                    if multiplier == 0 and prefix != text:
                        # Try recursive parsing for complex prefixes
                        multiplier = self.german_to_int(prefix)
                    if multiplier == 0:
                        multiplier = 1
                
                total += multiplier * scale_val
                remaining = suffix
                break
        
        # Handle remaining part
        if remaining and remaining != text:
            if remaining in self.german_ones:
                total += self.german_ones[remaining]
            elif remaining in self.german_tens:
                total += self.german_tens[remaining]
            else:
                # Try parsing remaining part
                remaining_val = self.german_to_int(remaining) if remaining != text else 0
                total += remaining_val
        
        return total if total > 0 else 0
    
    def chinese_to_int(self, text):
        """Convert Chinese numerals (traditional or simplified) to integer"""
        text = text.strip()
        
        # Simple digit mapping first
        if len(text) == 1 and text in self.chinese_digits:
            return self.chinese_digits[text]
        
        total = 0
        current = 0
        temp = 0
        
        i = 0
        while i < len(text):
            char = text[i]
            
            if char in self.chinese_digits:
                value = self.chinese_digits[char]
                
                if value < 10:  # Digit (0-9)
                    temp = value
                elif value == 10:  # 十
                    if temp == 0:
                        temp = 1
                    current += temp * 10
                    temp = 0
                elif value == 100:  # 百
                    if temp == 0:
                        temp = 1
                    current += temp * 100
                    temp = 0
                elif value == 1000:  # 千
                    if temp == 0:
                        temp = 1
                    current += temp * 1000
                    temp = 0
                elif value == 10000:  # 萬/万
                    if current == 0 and temp > 0:
                        current = temp
                        temp = 0
                    total += current * 10000
                    current = 0
                elif value == 100000000:  # 億/亿
                    if current == 0 and temp > 0:
                        current = temp
                        temp = 0
                    total += current * 100000000
                    current = 0
            
            i += 1
        
        # Add any remaining values
        if temp > 0:
            current += temp
        
        return total + current
    
    def get_language_priority(self, text):
        """Determine the priority based on language type"""
        # Priority order: Roman, English, Traditional Chinese, Simplified Chinese, German, Arabic
        if self.is_roman(text):
            return 0
        elif self.is_english(text):
            return 1
        elif self.is_traditional_chinese(text):
            return 2
        elif self.is_simplified_chinese(text):
            return 3
        elif self.is_german(text):
            return 4
        else:  # Arabic numerals
            return 5
    
    def is_roman(self, text):
        """Check if text is Roman numeral"""
        return bool(re.match(r'^[IVXLCDM]+$', text.upper()))
    
    def is_english(self, text):
        """Check if text is English number word"""
        words = text.lower().split()
        english_words = set(self.english_ones.keys()) | set(self.english_tens.keys()) | set(self.english_scales.keys())
        return len(words) > 0 and all(word in english_words for word in words)
    
    def is_traditional_chinese(self, text):
        """Check if text contains traditional Chinese characters"""
        traditional_chars = {'萬', '億'}
        simplified_chars = {'万', '亿'}
        
        has_traditional = any(char in traditional_chars for char in text)
        has_simplified = any(char in simplified_chars for char in text)
        
        return has_traditional and not has_simplified
    
    def is_simplified_chinese(self, text):
        """Check if text contains simplified Chinese characters"""
        simplified_chars = {'万', '亿'}
        chinese_chars = set(self.chinese_digits.keys())
        
        has_simplified = any(char in simplified_chars for char in text)
        is_chinese = all(char in chinese_chars for char in text) and len(text) > 0
        is_traditional = self.is_traditional_chinese(text)
        
        return has_simplified or (is_chinese and not is_traditional)
    
    def is_german(self, text):
        """Check if text is German number word"""
        text_lower = text.lower()
        
        parts = text_lower.split('und')
        german_words = set(self.german_ones.keys()) | set(self.german_tens.keys()) | set(self.german_scales.keys())
        
        for part in parts:
            if part in german_words:
                return True
            
            for german_word in german_words:
                if len(german_word) > 2 and german_word in part:
                    return True
        
        return False
    
    def parse_number(self, text):
        """Parse any supported number format to integer"""
        text = text.strip()
        
        # Try Arabic numeral first
        try:
            return int(text)
        except ValueError:
            pass
        
        # Try Roman numeral
        if self.is_roman(text):
            return self.roman_to_int(text)
        
        # Try English
        if self.is_english(text):
            return self.english_to_int(text)
        
        # Try Chinese
        if any(char in self.chinese_digits for char in text):
            return self.chinese_to_int(text)
        
        # Try German
        if self.is_german(text):
            return self.german_to_int(text)
        
        # If all else fails, return 0
        return 0

@app.route('/duolingo-sort', methods=['POST'])
def duolingo_sort():
    try:
        # Get JSON data - try multiple methods
        data = request.get_json()
        if not data:
            data = request.get_json(force=True)
        if not data and request.data:
            data = json.loads(request.data.decode('utf-8'))
        
        if not data:
            return jsonify({'error': 'Invalid input format'}), 400
        
        part = data.get('part')
        challenge_input = data.get('challengeInput', {})
        unsorted_list = challenge_input.get('unsortedList', [])
        
        if not part or not unsorted_list:
            return jsonify({'error': 'Missing required fields'}), 400
        
        parser = NumberParser()
        
        if part == "ONE":
            # Part 1: Convert all to integers and sort
            number_values = []
            for item in unsorted_list:
                value = parser.parse_number(item)
                number_values.append(value)
            
            # Sort and convert to strings
            sorted_numbers = sorted(number_values)
            result = [str(num) for num in sorted_numbers]
            
        elif part == "TWO":
            # Part 2: Sort by value, then by language priority
            number_pairs = []
            for item in unsorted_list:
                value = parser.parse_number(item)
                priority = parser.get_language_priority(item)
                number_pairs.append((value, priority, item))
            
            # Sort by value first, then by priority
            number_pairs.sort(key=lambda x: (x[0], x[1]))
            
            # Extract original representations
            result = [item[2] for item in number_pairs]
        
        else:
            return jsonify({'error': 'Invalid part specified'}), 400
        
        return jsonify({'sortedList': result})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500