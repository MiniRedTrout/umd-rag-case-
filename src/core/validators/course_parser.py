from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, RootModel, model_validator


class CourseContent(BaseModel):
    course_name: str = Field(description="Course name. Only in Russian. Type: string. Cannot be any other type.")
    content: str = Field(
        description="Answer about the course. Only in Russian. Type: string. Cannot be any other type."
    )

    @model_validator(mode="before")
    @classmethod
    def validate_course_name(cls, values: dict) -> dict:
        course_name = values.get("course_name")
        if not course_name or course_name.strip() == "":
            raise ValueError("Course name cannot be empty!")
        return values


class CourseList(RootModel):
    root: List[CourseContent] = Field(
        description="Courses list and the answers about them in Russian. Type: list. Cannot be any other type! Each element must be unique, there cannot be two or more elements with the same course_name!"
    )


course_parser = PydanticOutputParser(pydantic_object=CourseList)
