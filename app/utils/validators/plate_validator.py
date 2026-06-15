import re
from typing import Tuple, Set

# Regex patterns for different Vietnamese license plate formats
# Standard format examples:
# Civilian/Business (59A-123.45, 59AA-123.45, 59A-1234, 59-A1 123.45, etc.)
CIVILIAN_PATTERN = re.compile(r'^([1-9][0-9])([A-Z]{1,2})([0-9]{4,5})$')

# Military format: 2 letters + 2 digits + '-' + 2 digits or similar (e.g., KP1234, TM1234)
MILITARY_PATTERN = re.compile(r'^([A-Z]{2})([0-9]{4})$')

# Diplomatic/Foreign format: [Province Code]-[NG/QT/CV]-[Country Code]-[Serial]
# (e.g., 29-NG-123-45, 29-QT-123-45)
DIPLOMATIC_PATTERN = re.compile(r'^([1-9][0-9])(NG|QT|CV)([0-9]{5})$')

# Danh sách mã tỉnh/thành phố hợp lệ theo quy định Bộ Công An Việt Nam
# Nguồn: Thông tư 58/2020/TT-BCA, Nghị định 100/2019/NĐ-CP
VALID_PROVINCE_CODES: Set[int] = {
    # Hà Nội
    11, 12, 13, 14, 15, 16, 17, 18, 19, 40,
    # Các tỉnh phía Bắc
    20,  # Thái Nguyên
    21,  # Yên Bái
    22,  # Tuyên Quang
    23,  # Hà Giang
    24,  # Lào Cai
    25,  # Lai Châu
    26,  # Sơn La
    27,  # Điện Biên
    28,  # Hòa Bình
    29, 30, 31, 32, 33, 40,  # Hà Nội mở rộng
    34,  # Hải Dương
    35,  # Ninh Bình
    36,  # Thanh Hóa
    37,  # Nghệ An
    38,  # Hà Tĩnh
    39,  # Đồng Nai (cũ) / chuyển
    # Hải Phòng
    15, 16,
    # Vùng Đông Bắc & Tây Bắc
    41,  # TP HCM (phần mở rộng)
    42,  # (chưa sử dụng - dự phòng)
    43,  # Đà Nẵng
    # Miền Trung
    47,  # Đắk Lắk
    48,  # Đắk Nông
    49,  # Lâm Đồng
    50, 59,  # TP HCM
    51, 52, 53, 54, 55, 56, 57, 58,  # TP HCM mở rộng
    # Nam Bộ
    60,  # Đồng Nai
    61,  # Bình Dương
    62,  # Long An
    63,  # Tiền Giang
    64,  # Vĩnh Long
    65,  # Cần Thơ
    66,  # Đồng Tháp
    67,  # An Giang
    68,  # Kiên Giang
    69,  # Cà Mau
    70,  # Tây Ninh
    71,  # Bến Tre
    72,  # Bà Rịa - Vũng Tàu
    73,  # Quảng Bình
    74,  # Quảng Trị
    75,  # Thừa Thiên Huế
    76,  # Quảng Ngãi
    77,  # Bình Định
    78,  # Phú Yên
    79,  # Khánh Hòa
    80,  # Ninh Thuận (cũ)
    81,  # Gia Lai
    82,  # Kon Tum
    83,  # Sóc Trăng
    84,  # Trà Vinh
    85,  # Ninh Thuận
    86,  # Bình Thuận
    88,  # Vĩnh Phúc
    89,  # Hưng Yên
    90,  # Hà Nam
    92,  # Quảng Nam
    93,  # Bình Phước
    94,  # Bạc Liêu
    95,  # Hậu Giang
    97,  # Bắc Kạn
    98,  # Bắc Giang
    99,  # Bắc Ninh
}


def clean_plate_text(text: str) -> str:
    """Cleans raw OCR output for license plates.
    Removes special characters, spaces, and converts to uppercase.
    """
    if not text:
        return ""
    # Capitalize and remove non-alphanumeric characters
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    return cleaned


def format_plate_text(cleaned_text: str) -> str:
    """Formats clean license plate text into standard display format (e.g., 59A-123.45)."""
    # 1. Check diplomatic 29NG12345
    m = DIPLOMATIC_PATTERN.match(cleaned_text)
    if m:
        prov, code, num = m.groups()
        return f"{prov}-{code}-{num[:3]}-{num[3:]}"

    # 2. Check civilian 59A12345 or 59AA12345
    m = CIVILIAN_PATTERN.match(cleaned_text)
    if m:
        prov, char, num = m.groups()
        if len(num) == 5:
            return f"{prov}{char}-{num[:3]}.{num[3:]}"
        else:
            return f"{prov}{char}-{num}"

    # 3. Check military KP1234
    m = MILITARY_PATTERN.match(cleaned_text)
    if m:
        char, num = m.groups()
        return f"{char}-{num[:2]}-{num[2:]}"

    # Fallback formatting if it doesn't match standard patterns but starts with 2 digits
    if len(cleaned_text) >= 6 and cleaned_text[:2].isdigit():
        prov = cleaned_text[:2]
        rest = cleaned_text[2:]
        return f"{prov}-{rest}"
        
    return cleaned_text


def validate_plate(text: str) -> Tuple[bool, str]:
    """Validates Vietnamese license plates.
    
    Returns:
        Tuple[is_valid, validation_message]
    """
    cleaned = clean_plate_text(text)
    
    if not cleaned:
        return False, "Empty or invalid character set"
        
    if len(cleaned) < 5 or len(cleaned) > 10:
        return False, f"Length {len(cleaned)} is out of valid range (5-10)"

    # Helper to check province code validity
    def _check_province(code_str: str) -> str:
        """Returns a warning string if province code is not in standard table."""
        prov = int(code_str)
        if prov not in VALID_PROVINCE_CODES:
            return f" (Warning: Province code {code_str} may not be valid)"
        return ""

    m = CIVILIAN_PATTERN.match(cleaned)
    if m:
        prov_warning = _check_province(m.group(1))
        return True, f"Valid civilian plate{prov_warning}"
        
    if MILITARY_PATTERN.match(cleaned):
        return True, "Valid military plate"
        
    m = DIPLOMATIC_PATTERN.match(cleaned)
    if m:
        prov_warning = _check_province(m.group(1))
        return True, f"Valid diplomatic/foreign plate{prov_warning}"
        
    # Semi-valid fallback: Check if starts with a valid province code
    if cleaned[:2].isdigit():
        prov_code = int(cleaned[:2])
        if prov_code in VALID_PROVINCE_CODES:
            return True, "Heuristic match: starts with valid province code"
        elif 11 <= prov_code <= 99:
            return True, f"Heuristic match: starts with province code {prov_code} (unverified)"

    return False, "Does not match any known Vietnamese license plate formats"

