from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, RootModel, model_validator
from langchain_core.output_parsers import BaseOutputParser
import re


class QuestionContent(BaseModel):
    question: str = Field(description="Question to document text. Only in Russian. Type: string. Cannot be any other type.")


class RegexQuestionParser(BaseOutputParser):
    """Custom regex parser."""
    question_pattern = re.compile(r'"question": "(.*)"')

    def parse(self, text: str):
        text = re.findall(self.question_pattern, text)
        return QuestionContent(question=text[0])

    @property
    def _type(self) -> str:
        return "regex_output_parser"


regex_questions_parser = RegexQuestionParser()

json_questions_parser = PydanticOutputParser(pydantic_object=QuestionContent)
