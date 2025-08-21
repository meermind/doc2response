from pydantic import BaseModel, field_validator
from typing import Optional, List
from src.models.paths import OutputPaths
from src.models.metadata import load_course_metadata
import os
import json


class CourseContext(BaseModel):
    course_name: str
    module_name: str
    module_slug: Optional[str] = None
    output_base: str = "assistant_latex"
    input_base: str = "assistant_latex"
    list_top_k_skeleton: int = 500
    list_top_k_item: int = 50
    top_k: int = 20
    top_k_item: int = 10

    @field_validator("course_name", "module_name")
    def non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("course_name/module_name must not be empty")
        return v

    def output_paths(self) -> OutputPaths:
        return OutputPaths(base_dir=self.output_base, course_name=self.course_name, module_name=self.module_name)

    def module_filters(self) -> List[dict]:
        if self.module_slug:
            return [{"key": "module_slug", "value": self.module_slug, "operator": "=="}]
        return [{"key": "module_name", "value": self.module_name, "operator": "=="}]

    @staticmethod
    def from_json_str(s: str) -> "CourseContext":
        data = json.loads(s)
        return CourseContext(**data)

    @staticmethod
    def from_env(var_name: str = "D2R_CONTEXT") -> Optional["CourseContext"]:
        s = os.getenv(var_name)
        if not s:
            return None
        return CourseContext.from_json_str(s)

    @staticmethod
    def from_metadata_file(metadata_file: str, topic_index_1based: int, output_base: str = "assistant_latex", input_base: str = "assistant_latex", project_dir: Optional[str] = None) -> "CourseContext":
        course = load_course_metadata(metadata_file, project_dir)
        module = course.select_module_by_index(topic_index_1based)
        return CourseContext(
            course_name=course.course_name,
            module_name=module.module_name,
            module_slug=module.module_slug,
            output_base=output_base,
            input_base=input_base,
        )


