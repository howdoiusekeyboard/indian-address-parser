"""Hindi transliteration and script handling for multilingual addresses."""

import re


class HindiTransliterator:
    """
    Handles Hindi (Devanagari) to Latin transliteration and script detection.

    Supports:
    - Devanagari to Latin conversion
    - Common Hindi address terms
    - Mixed script (code-switched) addresses
    """

    # Devanagari Unicode range
    DEVANAGARI_START = 0x0900
    DEVANAGARI_END = 0x097F

    # Common Hindi address terms with transliterations
    HINDI_TERMS = {
        # Devanagari -> Latin
        'गली': 'GALI',
        'गलि': 'GALI',
        'मोहल्ला': 'MOHALLA',
        'नगर': 'NAGAR',
        'विहार': 'VIHAR',
        'पुरी': 'PURI',
        'पुर': 'PUR',
        'बाग': 'BAGH',
        'मार्ग': 'MARG',
        'रोड': 'ROAD',
        'मंजिल': 'FLOOR',
        'पहली': 'FIRST',
        'दूसरी': 'SECOND',
        'तीसरी': 'THIRD',
        'चौथी': 'FOURTH',
        'भूतल': 'GROUND FLOOR',
        'तहखाना': 'BASEMENT',
        'मकान': 'HOUSE',
        'प्लॉट': 'PLOT',
        'खसरा': 'KHASRA',
        'ब्लॉक': 'BLOCK',
        'सेक्टर': 'SECTOR',
        'कॉलोनी': 'COLONY',
        'इलाका': 'AREA',
        'क्षेत्र': 'AREA',
        'दिल्ली': 'DELHI',
        'नई दिल्ली': 'NEW DELHI',
        'नम्बर': 'NUMBER',
        'नंबर': 'NUMBER',
        'संख्या': 'NUMBER',
        'पास': 'NEAR',
        'सामने': 'OPPOSITE',
        'पीछे': 'BEHIND',
        'के पास': 'NEAR',
        'के सामने': 'OPPOSITE',
        'चौक': 'CHOWK',
        'बाजार': 'BAZAAR',
        'बस्ती': 'BASTI',
        'पार्क': 'PARK',
        'एक्सटेंशन': 'EXTENSION',
        'फेज': 'PHASE',
        'वार्ड': 'WARD',
        'जोन': 'ZONE',
    }

    # Devanagari consonants to Latin (basic ITRANS-like mapping)
    CONSONANT_MAP = {
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
        'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
        'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
        'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
        'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh',
        'ष': 'sh', 'स': 's', 'ह': 'h',
        'क़': 'q', 'ख़': 'kh', 'ग़': 'g', 'ज़': 'z', 'ड़': 'd',
        'ढ़': 'dh', 'फ़': 'f', 'य़': 'y',
    }

    # Devanagari vowels/matras
    VOWEL_MAP = {
        'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
        'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'अं': 'an', 'अः': 'ah',
        'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
        'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
        'ं': 'n', 'ः': 'h', '्': '',  # Halant (vowel killer)
        'ँ': 'n',  # Chandrabindu
    }

    # Devanagari digits
    DIGIT_MAP = {
        '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
        '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
    }

    def __init__(self, use_known_terms: bool = True):
        """
        Initialize transliterator.

        Args:
            use_known_terms: Use dictionary of known Hindi address terms
        """
        self.use_known_terms = use_known_terms

    def contains_devanagari(self, text: str) -> bool:
        """Check if text contains Devanagari script."""
        for char in text:
            code = ord(char)
            if self.DEVANAGARI_START <= code <= self.DEVANAGARI_END:
                return True
        return False

    def get_script_ratio(self, text: str) -> dict[str, float]:
        """
        Get ratio of different scripts in text.

        Returns dict with 'latin', 'devanagari', 'numeric', 'other' ratios.
        """
        if not text:
            return {'latin': 0.0, 'devanagari': 0.0, 'numeric': 0.0, 'other': 0.0}

        counts: dict[str, float] = {'latin': 0, 'devanagari': 0, 'numeric': 0, 'other': 0}
        total = 0

        for char in text:
            if char.isspace():
                continue
            total += 1
            code = ord(char)

            if self.DEVANAGARI_START <= code <= self.DEVANAGARI_END:
                counts['devanagari'] += 1
            elif char.isascii() and char.isalpha():
                counts['latin'] += 1
            elif char.isdigit():
                counts['numeric'] += 1
            else:
                counts['other'] += 1

        if total == 0:
            return counts

        return {k: v / total for k, v in counts.items()}

    def transliterate(self, text: str) -> str:
        """
        Transliterate Devanagari text to Latin script.

        Args:
            text: Input text (may be mixed script)

        Returns:
            Transliterated text in Latin script
        """
        if not self.contains_devanagari(text):
            return text

        # First, try to match known terms
        if self.use_known_terms:
            for hindi, latin in sorted(self.HINDI_TERMS.items(), key=lambda x: -len(x[0])):
                text = text.replace(hindi, f' {latin} ')

        # Then transliterate remaining Devanagari
        result = []
        i = 0
        while i < len(text):
            char = text[i]
            code = ord(char)

            if self.DEVANAGARI_START <= code <= self.DEVANAGARI_END:
                # Check digits first
                if char in self.DIGIT_MAP:
                    result.append(self.DIGIT_MAP[char])
                # Check vowels
                elif char in self.VOWEL_MAP:
                    result.append(self.VOWEL_MAP[char])
                # Check consonants
                elif char in self.CONSONANT_MAP:
                    result.append(self.CONSONANT_MAP[char])
                    # Add implicit 'a' unless followed by matra or halant
                    if i + 1 < len(text):
                        next_char = text[i + 1]
                        next_code = ord(next_char)
                        # If next is a matra (0x093E-0x094D) or halant, don't add 'a'
                        if not (0x093E <= next_code <= 0x094D):
                            result.append('a')
                    else:
                        result.append('a')
                else:
                    # Unknown Devanagari character
                    result.append(char)
            else:
                result.append(char)

            i += 1

        # Clean up
        output = ''.join(result)
        output = re.sub(r'\s+', ' ', output)
        return output.strip().upper()

    def normalize_mixed_script(self, text: str) -> str:
        """
        Handle code-mixed (Hindi + English) addresses.

        Transliterates Hindi portions while preserving English.
        """
        # Split on whitespace to handle word by word
        words = text.split()
        result = []

        for word in words:
            if self.contains_devanagari(word):
                # Check if it's a known term first
                if self.use_known_terms and word in self.HINDI_TERMS:
                    result.append(self.HINDI_TERMS[word])
                else:
                    result.append(self.transliterate(word))
            else:
                result.append(word.upper())

        return ' '.join(result)


def detect_language(text: str) -> str:
    """
    Simple language detection for address text.

    Returns: 'hindi', 'english', or 'mixed'
    """
    transliterator = HindiTransliterator()
    ratios = transliterator.get_script_ratio(text)

    if ratios['devanagari'] > 0.5:
        return 'hindi'
    elif ratios['latin'] > 0.5:
        return 'english'
    elif ratios['devanagari'] > 0 and ratios['latin'] > 0:
        return 'mixed'
    else:
        return 'english'
