import re
from typing import Tuple

# ISO 6346 standard character to value mapping table
# Note: Multiples of 11 (11, 22, 33) are omitted in the standard.
CHAR_VALUES = {
    'A': 10, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19,
    'J': 20, 'K': 21, 'L': 23, 'M': 24, 'N': 25, 'O': 26, 'P': 27, 'Q': 28, 'R': 29,
    'S': 30, 'T': 31, 'U': 32, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38
}

# Add values for digits 0-9
for i in range(10):
    CHAR_VALUES[str(i)] = i

# General container code pattern: 4 letters (last one is usually U, J, or Z) + 7 digits
CONTAINER_PATTERN = re.compile(r'^([A-Z]{4})([0-9]{7})$')


def _fix_owner_char(char: str) -> str:
    """Fix common OCR confusion for owner code characters (should be letters)."""
    mapping = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B'}
    return mapping.get(char, char)


def _fix_serial_char(char: str) -> str:
    """Fix common OCR confusion for serial number characters (should be digits)."""
    mapping = {'O': '0', 'Q': '0', 'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8'}
    return mapping.get(char, char)


def _apply_heuristics_11(cleaned: str) -> str:
    """Apply OCR heuristic fixes to a string of exactly 11 characters."""
    owner_part = cleaned[:4]
    serial_part = cleaned[4:10]
    check_digit = cleaned[10]

    fixed_owner = ''.join(_fix_owner_char(c) for c in owner_part)
    fixed_serial = ''.join(_fix_serial_char(c) for c in serial_part)
    fixed_check = _fix_serial_char(check_digit)

    return fixed_owner + fixed_serial + fixed_check


def clean_container_text(text: str) -> str:
    """Cleans OCR output for container codes.
    Removes whitespace, punctuation, converts to uppercase.
    Performs OCR error correction heuristics (e.g. replacing '0' with 'O' in owner code,
    or 'O' with '0' in serial numbers, '1' and 'I' swaps).
    Also handles 12-character OCR output by trying to truncate to 11.
    """
    if not text:
        return ""
    
    # Capitalize and remove non-alphanumeric
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    if len(cleaned) < 11:
        return cleaned
    
    if len(cleaned) == 11:
        return _apply_heuristics_11(cleaned)
    
    # Handle 12-character output: OCR often appends an extra digit at the end.
    # Try truncating to 11 characters and verify check digit.
    if len(cleaned) == 12:
        # Strategy 1: Drop the last character (most common OCR artifact)
        candidate_drop_last = _apply_heuristics_11(cleaned[:11])
        if CONTAINER_PATTERN.match(candidate_drop_last):
            try:
                expected = int(candidate_drop_last[10])
                calculated = calculate_check_digit(candidate_drop_last)
                if expected == calculated:
                    return candidate_drop_last
            except (ValueError, IndexError):
                pass
        
        # Strategy 2: Drop the extra digit at position 10 (before check digit)
        candidate_drop_mid = cleaned[:10] + cleaned[11]
        candidate_drop_mid = _apply_heuristics_11(candidate_drop_mid)
        if CONTAINER_PATTERN.match(candidate_drop_mid):
            try:
                expected = int(candidate_drop_mid[10])
                calculated = calculate_check_digit(candidate_drop_mid)
                if expected == calculated:
                    return candidate_drop_mid
            except (ValueError, IndexError):
                pass
        
        # Fallback: just take first 11 characters with heuristics
        return _apply_heuristics_11(cleaned[:11])
    
    # For strings longer than 12, just take first 11
    return _apply_heuristics_11(cleaned[:11])


def normalize_container_for_voting(text: str) -> str:
    """Normalizes a container code text for majority voting.
    Cleans the text and truncates to exactly 11 characters if it matches
    a container code pattern, so that near-identical OCR outputs
    (e.g. 'DFSU789456' vs 'DFSU7894560') are grouped together.
    """
    cleaned = clean_container_text(text)
    if len(cleaned) >= 11:
        candidate = cleaned[:11]
        if CONTAINER_PATTERN.match(candidate):
            return candidate
    return cleaned


def calculate_check_digit(container_code: str) -> int:
    """Calculates the check digit of a container code according to ISO 6346.
    
    Args:
        container_code: A string of at least 10 alphanumeric characters.
        
    Returns:
        The calculated check digit (0-9).
    """
    # Use only first 10 characters for calculation
    code = container_code[:10].upper()
    
    total = 0
    for idx, char in enumerate(code):
        if char not in CHAR_VALUES:
            raise ValueError(f"Invalid character '{char}' at index {idx} in ISO 6346 code")
        
        val = CHAR_VALUES[char]
        # Weight formula: 2^idx
        weight = 2 ** idx
        total += val * weight
        
    # ISO 6346 mod 11 calculation
    mod_val = total % 11
    
    # According to standard: if mod is 10, check digit is 0
    # (Or rather, the check digit cannot be 10, it wraps to 0 in practice)
    if mod_val == 10:
        return 0
    return mod_val


def validate_container_code(text: str) -> Tuple[bool, str]:
    """Validates container code format and check-digit validation.
    
    Returns:
        Tuple[is_valid, validation_message]
    """
    cleaned = clean_container_text(text)
    
    if len(cleaned) != 11:
        return False, f"Invalid length {len(cleaned)} (must be exactly 11 characters)"
        
    m = CONTAINER_PATTERN.match(cleaned)
    if not m:
        return False, f"Code '{cleaned}' does not match standard ISO 6346 format (4 Letters + 7 Digits)"
    
    # Check ISO 6346 Category Identifier (4th character)
    # U = Freight container, J = Detachable equipment, Z = Trailer/chassis
    category_id = cleaned[3]
    category_warning = ""
    if category_id not in ('U', 'J', 'Z'):
        category_warning = f" (Warning: Category identifier '{category_id}' is non-standard, expected U/J/Z)"
        
    # Extract expected check digit
    expected_digit = int(cleaned[10])
    
    try:
        calculated_digit = calculate_check_digit(cleaned)
    except ValueError as e:
        return False, str(e)
        
    if expected_digit == calculated_digit:
        return True, f"Valid ISO 6346 container code (Check digit matches){category_warning}"
    else:
        return False, f"Check digit mismatch: expected {expected_digit}, calculated {calculated_digit}{category_warning}"

