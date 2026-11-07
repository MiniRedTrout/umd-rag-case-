from typing import Final

CATALOG_CARD_TITLE_TEMPLATE: Final = "**{label}**"

CATALOG_CARD_PARENT_TEMPLATE: Final = "Входит в {parent}."

CATALOG_CARD_GROUPS_HEADER: Final = "Группы бизнес-процесса в категории:"

CATALOG_CARD_GROUP_ITEM_TEMPLATE: Final = "- {code} «{title}» (документов: {count})"

CATALOG_CARD_DOCUMENTS_TRUNCATED_TEMPLATE: Final = (
    "Показаны первые {shown} документов. Чтобы сузить список, запросите "
    "группу бизнес-процесса или начало шифра подробнее."
)

CATALOG_CARD_DOCUMENTS_UNAVAILABLE: Final = (
    "Список документов сейчас получить не удалось."
)

CATEGORY_NOT_FOUND_TEMPLATE: Final = "Категории бизнес-процесса {code} в базе знаний нет."

GROUP_NOT_FOUND_TEMPLATE: Final = "Группы бизнес-процесса {code} в базе знаний нет."

CATEGORY_GROUPS_TEMPLATE: Final = "В категорию бизнес-процесса {code} «{title}» входят группы:"

CIPHER_PREFIX_CONTEXT_TEMPLATE: Final = "Код {code} относится к {parent}."
