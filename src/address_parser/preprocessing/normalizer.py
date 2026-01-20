"""Address normalization utilities."""

import re


class AddressNormalizer:
    """
    Normalizes Indian addresses for consistent processing.

    Handles:
    - Case normalization
    - Whitespace cleanup
    - Common abbreviation expansion
    - Punctuation standardization
    - Number format standardization
    """

    # Common abbreviations in Indian addresses
    ABBREVIATIONS = {
        r'\bH\.?\s*NO\.?\b': 'HOUSE NO',
        r'\bH\.?\s*N\.?\b': 'HOUSE NO',
        r'\bHNO\.?\b': 'HOUSE NO',
        r'\bPLT\.?\s*NO\.?\b': 'PLOT NO',
        r'\bP\.?\s*NO\.?\b': 'PLOT NO',
        r'\bFL\.?\b': 'FLOOR',
        r'\bFLR\.?\b': 'FLOOR',
        r'\bGF\.?\b': 'GROUND FLOOR',
        r'\bFF\.?\b': 'FIRST FLOOR',
        r'\bSF\.?\b': 'SECOND FLOOR',
        r'\bTF\.?\b': 'THIRD FLOOR',
        r'\b1ST\s+FL\.?\b': 'FIRST FLOOR',
        r'\b2ND\s+FL\.?\b': 'SECOND FLOOR',
        r'\b3RD\s+FL\.?\b': 'THIRD FLOOR',
        r'\bGRD\.?\s*FL\.?\b': 'GROUND FLOOR',
        r'\bBLK\.?\b': 'BLOCK',
        r'\bBL\.?\b': 'BLOCK',
        r'\bSEC\.?\b': 'SECTOR',
        r'\bKH\.?\s*NO\.?\b': 'KHASRA NO',
        r'\bKHASRA\s*NO\.?\b': 'KHASRA NO',
        r'\bKH\.?\b': 'KHASRA',
        r'\bCOL\.?\b': 'COLONY',
        r'\bNGR\.?\b': 'NAGAR',
        r'\bMKT\.?\b': 'MARKET',
        r'\bRD\.?\b': 'ROAD',
        r'\bST\.?\b': 'STREET',
        r'\bLN\.?\b': 'LANE',
        r'\bEXTN\.?\b': 'EXTENSION',
        r'\bEXT\.?\b': 'EXTENSION',
        r'\bPH\.?\b': 'PHASE',
        r'\bNR\.?\b': 'NEAR',
        r'\bOPP\.?\b': 'OPPOSITE',
        r'\bBHD\.?\b': 'BEHIND',
        r'\bADJ\.?\b': 'ADJACENT',
        r'\bWZ\.?\b': 'WZ',  # West Zone
        r'\bEZ\.?\b': 'EZ',  # East Zone
        r'\bNZ\.?\b': 'NZ',  # North Zone
        r'\bSZ\.?\b': 'SZ',  # South Zone
        r'\bDL\.?\b': 'DELHI',
        r'\bN\.?\s*DELHI\b': 'NEW DELHI',
    }

    # Floor name patterns
    FLOOR_PATTERNS = {
        r'\bGROUND\b': 'GROUND',
        r'\bBASEMENT\b': 'BASEMENT',
        r'\bFIRST\b': 'FIRST',
        r'\bSECOND\b': 'SECOND',
        r'\bTHIRD\b': 'THIRD',
        r'\bFOURTH\b': 'FOURTH',
        r'\bFIFTH\b': 'FIFTH',
        r'\b1ST\b': 'FIRST',
        r'\b2ND\b': 'SECOND',
        r'\b3RD\b': 'THIRD',
        r'\b4TH\b': 'FOURTH',
        r'\b5TH\b': 'FIFTH',
    }

    def __init__(self, uppercase: bool = True, expand_abbrev: bool = True):
        """
        Initialize normalizer.

        Args:
            uppercase: Convert text to uppercase
            expand_abbrev: Expand common abbreviations
        """
        self.uppercase = uppercase
        self.expand_abbrev = expand_abbrev

        # Compile regex patterns
        self._abbrev_patterns = {
            re.compile(pattern, re.IGNORECASE): replacement
            for pattern, replacement in self.ABBREVIATIONS.items()
        }

    def normalize(self, address: str) -> str:
        """
        Normalize an address string.

        Args:
            address: Raw address string

        Returns:
            Normalized address string
        """
        if not address:
            return ""

        text = address

        # Basic cleanup
        text = self._clean_whitespace(text)
        text = self._standardize_punctuation(text)

        # Expand abbreviations
        if self.expand_abbrev:
            text = self._expand_abbreviations(text)

        # Case normalization
        if self.uppercase:
            text = text.upper()

        # Final whitespace cleanup
        text = self._clean_whitespace(text)

        return text

    def _clean_whitespace(self, text: str) -> str:
        """Remove extra whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        # Remove spaces around punctuation
        text = re.sub(r'\s*,\s*', ', ', text)
        text = re.sub(r'\s*-\s*', '-', text)
        # Trim
        return text.strip()

    def _standardize_punctuation(self, text: str) -> str:
        """Standardize punctuation marks."""
        # Replace various dash types with standard hyphen
        text = re.sub(r'[–—]', '-', text)
        # Remove duplicate punctuation
        text = re.sub(r',+', ',', text)
        text = re.sub(r'-+', '-', text)
        # Remove trailing punctuation before comma
        text = re.sub(r'-,', ',', text)
        return text

    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        for pattern, replacement in self._abbrev_patterns.items():
            text = pattern.sub(replacement, text)
        return text

    def extract_pincode(self, address: str) -> str | None:
        """Extract 6-digit Indian PIN code from address."""
        match = re.search(r'\b[1-9]\d{5}\b', address)
        return match.group(0) if match else None

    def remove_pincode(self, address: str) -> str:
        """Remove PIN code from address."""
        return re.sub(r'\b[1-9]\d{5}\b', '', address)

    def tokenize(self, text: str) -> list[str]:
        """
        Simple tokenization preserving address-specific patterns.

        Args:
            text: Normalized address text

        Returns:
            List of tokens
        """
        # Split on whitespace but keep special patterns together
        # e.g., "H-3" stays as one token, "110041" stays together
        tokens = []

        # Pattern to match address tokens
        pattern = r'''
            [A-Z0-9]+[-/][A-Z0-9/]+  |  # Compound identifiers like H-3, 24/1/3
            [A-Z]+\d+               |  # Letter+number combos like A5
            \d+[A-Z]+               |  # Number+letter combos like 5A
            [A-Z]+                  |  # Words
            \d+                     |  # Numbers
            [,.]                       # Punctuation
        '''

        for match in re.finditer(pattern, text.upper(), re.VERBOSE):
            token = match.group(0)
            if token.strip():
                tokens.append(token)

        return tokens
