from typing import Final


DOCUMENT_ANSWER_TEMPLATE: Final = """Нашёл документ по вашему запросу:

{item}"""

DOCUMENTS_ANSWER_TEMPLATE: Final = """Нашёл документы по вашему запросу:

{items}"""

DOCUMENT_WITHOUT_LINK_TEMPLATE: Final = "{title} (ссылка на документ не указана)"

DOCUMENT_NOT_FOUND_TEXT: Final = "Документ не найден."


DOCUMENTS_BY_CODE_TEMPLATE: Final = """Документы с шифром, начинающимся на {code}:

{items}"""
 
DOCUMENTS_BY_CODE_NOT_FOUND_TEXT: Final = "Документов с шифром, начинающимся на {code}, в базе знаний нет."