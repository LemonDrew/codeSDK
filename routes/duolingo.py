from flask import request, jsonify
import re
import logging

from routes import app

logger = logging.getLogger(__name__)

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
                
                if ones_val and tens_val:
                    return tens_val + ones_val
        
        # Handle simple cases
        if text in self.german_ones:
            return self.german_ones[text]
        if text in self.german_tens:
            return self.german_tens[text]
        
        # Handle compound words like "dreihundertelf" (311)
        total = 0
        remaining = text
        
        # Check for scales
        for scale_word, scale_val in self.german_scales.items():
            if scale_word in remaining:
                prefix = remaining.split(scale_word)[0]
                suffix = remaining.split(scale_word)[1] if scale_word in remaining else ""
                
                multiplier = 1
                if prefix:
                    multiplier = self.german_ones.get(prefix, 1)
                
                total += multiplier * scale_val
                remaining = suffix
        
        # Handle remaining part
        if remaining in self.german_ones:
            total += self.german_ones[remaining]
        elif remaining in self.german_tens:
            total += self.german_tens[remaining]
        
        return total
    
    def chinese_to_int(self, text):
        """Convert Chinese numerals (traditional or simplified) to integer"""
        text = text.strip()
        
        # Simple digit mapping first
        if len(text) == 1 and text in self.chinese_digits:
            return self.chinese_digits[text]
        
        total = 0
        current = 0
        
        i = 0
        while i < len(text):
            char = text[i]
            
            if char in self.chinese_digits:
                value = self.chinese_digits[char]
                
                if value < 10:  # Digit
                    current = current * 10 + value if current > 0 else value
                elif value == 10:  # 十
                    if current == 0:
                        current = 1
                    current *= 10
                elif value == 100:  # 百
                    if current == 0:
                        current = 1
                    current *= 100
                elif value == 1000:  # 千
                    if current == 0:
                        current = 1
                    current *= 1000
                elif value == 10000:  # 萬/万
                    total += current * 10000
                    current = 0
                elif value == 100000000:  # 億/亿
                    total += current * 100000000
                    current = 0
            
            i += 1
        
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
        return all(word in english_words for word in words)
    
    def is_traditional_chinese(self, text):
        """Check if text contains traditional Chinese characters"""
        traditional_chars = {'萬', '億'}
        return any(char in traditional_chars for char in text)
    
    def is_simplified_chinese(self, text):
        """Check if text contains simplified Chinese characters"""
        simplified_chars = {'万', '亿'}
        chinese_chars = set(self.chinese_digits.keys())
        return (any(char in simplified_chars for char in text) or 
                all(char in chinese_chars for char in text))
    
    def is_german(self, text):
        """Check if text is German number word"""
        words = text.lower().split('und')
        german_words = set(self.german_ones.keys()) | set(self.german_tens.keys()) | set(self.german_scales.keys())
        
        # Check compound words
        if len(words) == 1:
            word = words[0]
            # Check if it's a compound word containing German number parts
            for german_word in german_words:
                if german_word in word:
                    return True
        
        return any(word in german_words for word in words)
    
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
        
        # Try German
        if self.is_german(text):
            return self.german_to_int(text)
        
        # Try Chinese (both traditional and simplified)
        if any(char in self.chinese_digits for char in text):
            return self.chinese_to_int(text)
        
        # If all else fails, return 0
        return 0

@app.route('/duolingo-sort', methods=['POST'])
def duolingo_sort():
    try:
        data = request.get_json()
        
        if not data or 'part' not in data or 'challengeInput' not in data:
            return jsonify({'error': 'Invalid input format'}), 400
        
        part = data['part']
        unsorted_list = data['challengeInput']['unsortedList']
        
        parser = NumberParser()
        
        if part == "ONE":
            # Part 1: Convert all to integers and sort
            number_pairs = []
            for item in unsorted_list:
                value = parser.parse_number(item)
                number_pairs.append(value)
            
            # Sort and convert to strings
            sorted_numbers = sorted(number_pairs)
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)