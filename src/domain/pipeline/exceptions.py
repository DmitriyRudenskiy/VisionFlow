class PipelineDomainException(Exception):
    """Базовое исключение для домена Pipeline"""
    pass

class InvalidStepStateTransition(PipelineDomainException):
    """Ошибка перехода между статусами шага"""
    pass

class StepNotFoundError(PipelineDomainException):
    """Шаг не найден в пайплайне"""
    pass