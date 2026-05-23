class ImageDomainException(Exception):
    """Базовое исключение домена Image"""
    pass

class InvalidImageFormat(ImageDomainException):
    """Неподдерживаемый формат изображения"""
    pass

class InvalidImageStateTransition(ImageDomainException):
    """Некорректный переход статуса изображения"""
    pass