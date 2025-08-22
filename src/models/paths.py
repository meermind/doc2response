from pydantic import BaseModel, field_validator
import os


class OutputPaths(BaseModel):
    base_dir: str = "assistant_latex"
    course_name: str
    module_name: str

    @field_validator("course_name", "module_name")
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("course_name/module_name must not be empty")
        return v

    def module_dir(self) -> str:
        return os.path.join(self.base_dir, self.course_name, self.module_name)

    def intro_path(self, title: str = "Introduction") -> str:
        return os.path.join(self.module_dir(), f"{title}.tex")

    def subsection_path(self, title: str) -> str:
        return os.path.join(self.module_dir(), f"{title}.tex")



