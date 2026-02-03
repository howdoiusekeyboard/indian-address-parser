"""Delhi locality gazetteer for fuzzy matching and validation."""

from rapidfuzz import fuzz, process


class DelhiGazetteer:
    """
    Gazetteer of Delhi localities, areas, and common address terms.

    Used for:
    - Fuzzy matching to correct misspellings
    - Entity validation
    - Confidence boosting for known locations
    """

    # Major Delhi localities/areas
    LOCALITIES = {
        # South Delhi
        "SAKET", "MALVIYA NAGAR", "HAUZ KHAS", "GREEN PARK", "GREATER KAILASH",
        "DEFENCE COLONY", "LAJPAT NAGAR", "SOUTH EXTENSION", "CHITTARANJAN PARK",
        "KALKAJI", "NEHRU PLACE", "OKHLA", "JASOLA", "SARITA VIHAR",
        "ALAKNANDA", "SAFDARJUNG", "VASANT KUNJ", "MEHRAULI", "CHATTARPUR",

        # North Delhi
        "CIVIL LINES", "MODEL TOWN", "MUKHERJEE NAGAR", "KAMLA NAGAR",
        "SHAKTI NAGAR", "GULABI BAGH", "ASHOK VIHAR", "SHALIMAR BAGH",
        "PITAMPURA", "ROHINI", "NARELA", "BAWANA", "ALIPUR",

        # East Delhi
        "PREET VIHAR", "MAYUR VIHAR", "PATPARGANJ", "PANDAV NAGAR",
        "LAKSHMI NAGAR", "SHAKARPUR", "GEETA COLONY", "GANDHI NAGAR",
        "DILSHAD GARDEN", "SEELAMPUR", "SHAHDARA", "ANAND VIHAR",

        # West Delhi
        "JANAKPURI", "DWARKA", "PALAM", "UTTAM NAGAR", "VIKASPURI",
        "TILAK NAGAR", "RAJOURI GARDEN", "PUNJABI BAGH", "PASCHIM VIHAR",
        "MEERA BAGH", "PEERAGARHI", "MUNDKA", "NANGLOI", "NAJAFGARH",
        "BINDAPUR", "KAKROLA", "MOHAN GARDEN", "NAWADA",

        # Central Delhi
        "CONNAUGHT PLACE", "KAROL BAGH", "PAHARGANJ", "DARYAGANJ",
        "CHANDNI CHOWK", "SADAR BAZAAR", "RAJENDER NAGAR", "PATEL NAGAR",
        "KIRTI NAGAR", "MOTIA KHAN", "ANAND PARBAT", "JHANDEWALAN",

        # New Delhi
        "CHANAKYAPURI", "LODHI ROAD", "GOLF LINKS", "JORBAGH",
        "SUNDAR NAGAR", "NIZAMUDDIN", "LODI COLONY", "PANDARA ROAD",

        # Other areas
        "BADARPUR", "TUGHLAKABAD", "SANGAM VIHAR", "MADANPUR KHADAR",
        "GOVINDPURI", "AMBEDKAR NAGAR", "LADO SARAI", "TIGRI",
        "BURARI", "KARAWAL NAGAR", "BHAJANPURA", "MUSTAFABAD",
        "JAFFRABAD", "MAUJPUR", "GOKALPUR", "SEEMAPURI",
    }

    # Common colony/nagar suffixes
    NAGAR_SUFFIXES = {
        "NAGAR", "VIHAR", "COLONY", "ENCLAVE", "EXTENSION", "PURI",
        "PARK", "GARDEN", "BAGH", "KUNJ", "APARTMENT", "RESIDENCY",
        "COMPLEX", "PHASE", "SECTOR", "BLOCK", "POCKET",
    }

    # Common area names from the training data
    COMMON_AREAS = {
        "KAUNWAR SINGH NAGAR", "BABA HARI DAS COLONY", "TIKARI KALA",
        "CHANCHAL PARK", "SWARN PARK", "MUNDKA", "NANGLOI", "BAKKARWALA",
        "MAJRA DABAS", "CHAND NAGAR", "RANHOLA", "BAPROLA", "POOTH KHURD",
        "KIRARI", "SULTANPURI", "MANGOLPURI", "BEGUMPUR", "KADIPUR",
        "RAMA VIHAR", "PREM NAGAR", "VIJAY PARK", "AMBICA VIHAR",
        "SHIV PURI", "BUDH VIHAR", "POOTH KALAN", "QUTUBGARH",
        "RANI KHERA", "SHAHABAD DAIRY", "SAMAIPUR", "JAHANGIRPURI",
        "SANNOTH", "KANJHAWALA", "BAWANA", "ALIPUR",
    }

    # Common Hindi transliterated terms
    HINDI_TERMS = {
        "MOHALLA", "GALI", "KATRA", "BASTI", "BAZAR", "CHOWK",
        "GANJ", "PUR", "ABAD", "GARH", "GAON", "KHERA", "KHURD", "KALAN",
    }

    def __init__(self, min_similarity: float = 80.0):
        """
        Initialize gazetteer.

        Args:
            min_similarity: Minimum fuzzy match score (0-100)
        """
        self.min_similarity = min_similarity

        # Build combined set for matching
        self.all_places = (
            self.LOCALITIES |
            self.COMMON_AREAS |
            {f"{term}" for term in self.HINDI_TERMS}
        )

    def fuzzy_match(
        self,
        text: str,
        limit: int = 3
    ) -> list[tuple[str, float]]:
        """
        Find fuzzy matches for a text in the gazetteer.

        Args:
            text: Text to match
            limit: Maximum number of matches

        Returns:
            List of (matched_text, score) tuples
        """
        if not text or len(text) < 3:
            return []

        matches = process.extract(
            text.upper(),
            self.all_places,
            scorer=fuzz.ratio,
            limit=limit
        )

        return [(m[0], m[1]) for m in matches if m[1] >= self.min_similarity]

    def is_known_locality(self, text: str, threshold: float = 85.0) -> bool:
        """Check if text matches a known locality."""
        matches = self.fuzzy_match(text, limit=1)
        return bool(matches and matches[0][1] >= threshold)

    def correct_spelling(self, text: str) -> str | None:
        """
        Attempt to correct spelling using gazetteer.

        Returns corrected text or None if no good match.
        """
        matches = self.fuzzy_match(text, limit=1)
        if matches and matches[0][1] >= 90.0:
            return matches[0][0]
        return None

    def get_locality_type(self, text: str) -> str | None:
        """
        Determine if text contains a known locality type suffix.

        Returns the suffix type or None.
        """
        text_upper = text.upper()
        for suffix in self.NAGAR_SUFFIXES:
            if text_upper.endswith(suffix):
                return suffix
        return None

    def validate_pincode(self, pincode: str, locality: str | None = None) -> bool:
        """
        Validate if a pincode is valid for Delhi.

        Delhi pincodes are in range 110001-110097.
        """
        if not pincode or not pincode.isdigit() or len(pincode) != 6:
            return False

        code = int(pincode)
        # Delhi pincode range
        return 110001 <= code <= 110097
