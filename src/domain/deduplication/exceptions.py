class DeduplicationDomainException(Exception):
    """Базовое исключение для домена Deduplication"""
    pass

class InvalidSimilarityScore(DeduplicationDomainException):
    """Ошибка валидации Score"""
    pass