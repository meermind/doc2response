import os
from typing import List, Dict, Tuple


class MetadataStore:
    @staticmethod
    def metadata_dir(module_dir: str) -> str:
        return os.path.join(module_dir, "metadata")
    @staticmethod
    def load_sections(input_base_dir: str, course: str, module_name: str) -> Tuple[List[Dict[str, str]], str]:
        import json as _json
        module_dir = os.path.join(input_base_dir, course, module_name)
        metadata_path = os.path.join(MetadataStore.metadata_dir(module_dir), "metadata.json")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            meta = _json.load(f)
        sections = meta.get('sections', [])
        enhanced_dir = os.path.join(module_dir, 'enhanced')
        # Prefer enhanced subsection file if present
        for s in sections:
            p = s.get('path')
            if not p:
                continue
            if s.get('type') == 'subsection':
                candidate = os.path.join(enhanced_dir, os.path.basename(p))
                if os.path.exists(candidate):
                    s['path'] = candidate
        # Find intro (first section)
        intro_path = ""
        for s in sections:
            if s.get('type') == 'section':
                intro_path = s.get('path', '')
                break
        return sections, intro_path

    @staticmethod
    def save_sections(module_dir: str, sections: List[Dict[str, str]]) -> None:
        import json as _json
        meta_dir = MetadataStore.metadata_dir(module_dir)
        os.makedirs(meta_dir, exist_ok=True)
        meta_path = os.path.join(meta_dir, "metadata.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            _json.dump({"sections": sections}, f, indent=2)

    @staticmethod
    def module_dir(base_dir: str, course: str, module_name: str) -> str:
        return os.path.join(base_dir, course, module_name)

    @staticmethod
    def save_unique_topics(base_dir: str, course: str, module_name: str, topics: List[str]) -> str:
        import json as _json
        mdir = MetadataStore.module_dir(base_dir, course, module_name)
        meta_dir = MetadataStore.metadata_dir(mdir)
        os.makedirs(meta_dir, exist_ok=True)
        path = os.path.join(meta_dir, "unique_topics.json")
        with open(path, 'w', encoding='utf-8') as f:
            _json.dump({"module": module_name, "unique_topics": topics}, f, indent=2)
        return path

    @staticmethod
    def save_mapping(out_dir: str, mapping: Dict[str, any]) -> str:
        import json as _json
        meta_dir = MetadataStore.metadata_dir(out_dir)
        os.makedirs(meta_dir, exist_ok=True)
        path = os.path.join(meta_dir, "subsection_mapping.json")
        with open(path, 'w', encoding='utf-8') as f:
            _json.dump(mapping, f, indent=2)
        return path

    @staticmethod
    def save_skeleton(out_dir: str, intro_title: str, subsections: List[Dict[str, any]]) -> str:
        import json as _json
        meta_dir = MetadataStore.metadata_dir(out_dir)
        os.makedirs(meta_dir, exist_ok=True)
        path = os.path.join(meta_dir, "skeleton.json")
        with open(path, 'w', encoding='utf-8') as f:
            _json.dump({"intro_title": intro_title, "subsections": subsections}, f, indent=2)
        return path

    @staticmethod
    def load_mapping(out_dir: str) -> Dict[str, any]:
        import json as _json
        path = os.path.join(MetadataStore.metadata_dir(out_dir), "subsection_mapping.json")
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return _json.load(f)

    @staticmethod
    def load_skeleton(out_dir: str) -> Dict[str, any]:
        import json as _json
        path = os.path.join(MetadataStore.metadata_dir(out_dir), "skeleton.json")
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return _json.load(f)


