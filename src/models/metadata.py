from __future__ import annotations

import json
import os
from typing import List, Optional, Iterator

from pydantic import BaseModel, field_validator


class ContentRef(BaseModel):
    content_type: str
    path: str

    @field_validator("content_type")
    def non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("content_type must not be empty")
        return v


class Item(BaseModel):
    name: str
    transformed_slug: str
    content: List[ContentRef] = []


class Lesson(BaseModel):
    lesson_name: str
    lesson_slug: str
    items: List[Item] = []


class ModuleMeta(BaseModel):
    module_name: str
    module_slug: str
    lessons: List[Lesson] = []


class CourseMeta(BaseModel):
    course_name: str
    course_slug: str
    modules: List[ModuleMeta] = []

    def resolve_paths(self, project_dir: str) -> None:
        for module in self.modules:
            for lesson in module.lessons:
                for item in lesson.items:
                    for c in item.content:
                        if not c.path:
                            continue
                        if os.path.isabs(c.path) or os.path.exists(c.path):
                            continue
                        c.path = os.path.join(project_dir, c.path)

    def select_module_by_index(self, topic_index_1based: int) -> ModuleMeta:
        if topic_index_1based < 1 or topic_index_1based > len(self.modules):
            raise IndexError("topic index out of range")
        return self.modules[topic_index_1based - 1]

    def iter_transcript_paths(self, module: ModuleMeta) -> Iterator[str]:
        for lesson in module.lessons:
            for item in lesson.items:
                for c in item.content:
                    if c.content_type == "transcript" and c.path and c.path.endswith(".txt"):
                        yield c.path


def load_course_metadata(metadata_file: str, project_dir: Optional[str] = None) -> CourseMeta:
    with open(metadata_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    course = CourseMeta.model_validate(data)
    if project_dir:
        course.resolve_paths(project_dir)
    return course

