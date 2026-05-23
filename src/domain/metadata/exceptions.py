class MetadataDomainException(Exception):
    """Базовое исключение для домена Metadata"""
    pass

class InvalidScoreRange(MetadataDomainException):
    """Значение выходит за пределы 0.0-1.0"""
    pass