"""
Schema Validator for Generated Payloads.
Validates synthetic patient, trajectory, mutation, and scenario instances against draft-07 JSON Schemas.
"""
import os
import json
import jsonschema
from pathlib import Path
from typing import Dict, Any, Tuple, List

HOSPITAL_ROOT = Path("c:/Users/shyam/OneDrive/Documents/HOSPITAL")

class SchemaValidator:
    SCHEMAS_DIR = HOSPITAL_ROOT / "STAGE_05_GENAI" / "schemas"

    def __init__(self):
        self._schemas: Dict[str, Dict[str, Any]] = {}
        self._load_all_schemas()

    def _load_all_schemas(self):
        if self.SCHEMAS_DIR.exists():
            for f in self.SCHEMAS_DIR.glob("*.json"):
                with open(f, "r", encoding="utf-8") as sf:
                    try:
                        self._schemas[f.name] = json.load(sf)
                    except Exception as e:
                        print(f"Error loading schema {f.name}: {e}")

    def validate_instance(self, instance: Dict[str, Any], schema_name: str) -> Tuple[bool, List[str]]:
        if schema_name not in self._schemas:
            # Try loading directly
            s_path = self.SCHEMAS_DIR / schema_name
            if s_path.exists():
                with open(s_path, "r", encoding="utf-8") as sf:
                    self._schemas[schema_name] = json.load(sf)
            else:
                return False, [f"Schema '{schema_name}' not found in {self.SCHEMAS_DIR}"]

        schema = self._schemas[schema_name]
        errors = []
        try:
            # We construct a local resolver so $ref references between schemas work seamlessly
            resolver = jsonschema.RefResolver(
                base_uri=f"{self.SCHEMAS_DIR.as_uri()}/",
                referrer=schema,
                store={k: v for k, v in self._schemas.items()}
            )
            validator = jsonschema.Draft7Validator(schema, resolver=resolver)
            for err in validator.iter_errors(instance):
                errors.append(f"Validation Error at '{'.'.join(str(p) for p in err.path)}': {err.message}")
        except Exception as e:
            errors.append(f"Schema validation internal error: {str(e)}")

        return len(errors) == 0, errors

if __name__ == "__main__":
    sv = SchemaValidator()
    print("Loaded schemas:", list(sv._schemas.keys()))
