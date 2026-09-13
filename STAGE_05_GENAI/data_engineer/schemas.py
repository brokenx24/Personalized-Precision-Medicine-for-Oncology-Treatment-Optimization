"""
Data Engineer Schemas & Validation.
Defines Draft-07 JSON schemas for all Data Engineer outputs and provides validation utilities.
"""
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import jsonschema

try:
    from data_engineer.path_resolver import resolve_stage5_path
except ImportError:
    from path_resolver import resolve_stage5_path

SCHEMAS: Dict[str, Dict[str, Any]] = {
    "source_manifest": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "SourceManifest",
        "type": "object",
        "required": ["timestamp", "sources", "summary"],
        "properties": {
            "timestamp": {"type": "string"},
            "sources": {"type": "object"},
            "summary": {
                "type": "object",
                "required": ["total_checked", "available_count", "not_available_count"],
                "properties": {
                    "total_checked": {"type": "integer"},
                    "available_count": {"type": "integer"},
                    "not_available_count": {"type": "integer"}
                }
            }
        }
    },
    "cleaning_report": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "CleaningReport",
        "type": "object",
        "required": ["timestamp", "initial_rows", "final_clean_rows", "exact_duplicates_removed", "total_missing_values_imputed", "total_outliers_clipped", "transformations"],
        "properties": {
            "timestamp": {"type": "string"},
            "initial_rows": {"type": "integer", "minimum": 0},
            "final_clean_rows": {"type": "integer", "minimum": 0},
            "exact_duplicates_removed": {"type": "integer", "minimum": 0},
            "total_rows_dropped": {"type": "integer", "minimum": 0},
            "total_missing_values_imputed": {"type": "integer", "minimum": 0},
            "total_outliers_clipped": {"type": "integer", "minimum": 0},
            "imputed_columns": {"type": "object"},
            "clipped_outliers": {"type": "object"},
            "transformations": {"type": "array"}
        }
    },
    "validation_report": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "ValidationReport",
        "type": "object",
        "required": ["total_records_checked", "missing_required_columns", "null_value_violations", "validation_status"],
        "properties": {
            "total_records_checked": {"type": "integer", "minimum": 0},
            "missing_required_columns": {"type": "array", "items": {"type": "string"}},
            "null_value_violations": {"type": "integer", "minimum": 0},
            "boundary_violations": {"type": "integer", "minimum": 0},
            "validation_status": {"type": "string", "enum": ["PASS", "FAIL"]}
        }
    },
    "privacy_report": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "PrivacyReport",
        "type": "object",
        "required": ["timestamp", "records_audited", "record_type", "direct_identifiers_found", "direct_identifiers_removed", "de_identification_verified", "full_anonymization_guarantee", "privacy_status"],
        "properties": {
            "timestamp": {"type": "string"},
            "records_audited": {"type": "integer", "minimum": 0},
            "record_type": {"type": "string", "enum": ["REFERENCE", "HISTORICAL", "SYNTHETIC"]},
            "direct_identifiers_found": {"type": "integer", "minimum": 0},
            "direct_identifiers_removed": {"type": "boolean"},
            "privacy_violations": {"type": "array"},
            "de_identification_verified": {"type": "boolean"},
            "full_anonymization_guarantee": {"type": "string"},
            "anonymization_notes": {"type": "string"},
            "privacy_status": {"type": "string", "enum": ["PASS", "FAIL"]}
        }
    },
    "data_manifest": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "DataManifest",
        "type": "object",
        "required": ["generated_at", "total_files", "files", "environment"],
        "properties": {
            "generated_at": {"type": "string"},
            "total_files": {"type": "integer"},
            "environment": {
                "type": "object",
                "required": ["python_version", "random_seed"],
                "properties": {
                    "python_version": {"type": "string"},
                    "random_seed": {"type": "integer"}
                }
            },
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["file", "type", "size_bytes", "sha256", "status"],
                    "properties": {
                        "file": {"type": "string"},
                        "type": {"type": "string"},
                        "size_bytes": {"type": "integer"},
                        "sha256": {"type": "string"},
                        "status": {"type": "string"}
                    }
                }
            }
        }
    },
    "mutation_frequency": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "MutationFrequency",
        "type": "object",
        "required": ["metadata", "cancer_specific_mutations"],
        "properties": {
            "metadata": {"type": "object"},
            "cancer_specific_mutations": {"type": "object"}
        }
    },
    "mutation_cooccurrence": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "MutationCooccurrence",
        "type": "object",
        "required": ["metadata", "pairwise_relationships"],
        "properties": {
            "metadata": {"type": "object"},
            "pairwise_relationships": {"type": "object"}
        }
    },
    "trajectory_statistics": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "TrajectoryStatistics",
        "type": "object",
        "required": ["metadata", "longitudinal_data_status", "recist_transition_probabilities"],
        "properties": {
            "metadata": {"type": "object"},
            "longitudinal_data_status": {"type": "string"},
            "recist_transition_probabilities": {"type": "object"},
            "biomarker_drift_parameters": {"type": "object"}
        }
    },
    "clinical_distribution": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "ClinicalDistribution",
        "type": "object",
        "required": ["metadata", "cancer_type_proportions", "demographics"],
        "properties": {
            "metadata": {"type": "object"},
            "cancer_type_proportions": {"type": "object"},
            "demographics": {"type": "object"}
        }
    },
    "biomarker_distribution": {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "BiomarkerDistribution",
        "type": "object",
        "required": ["metadata", "laboratory_distributions"],
        "properties": {
            "metadata": {"type": "object"},
            "laboratory_distributions": {"type": "object"}
        }
    }
}

def export_schemas(output_dir: Path = None) -> List[Path]:
    """Export all JSON schemas to disk under STAGE_05_GENAI/data/schemas/."""
    if output_dir is None:
        output_dir = resolve_stage5_path("data/schemas")
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    for name, schema_dict in SCHEMAS.items():
        file_path = output_dir / f"{name}.schema.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(schema_dict, f, indent=2)
        saved_files.append(file_path)
    return saved_files

def validate_data_schema(data: Dict[str, Any], schema_name: str) -> Tuple[bool, List[str]]:
    """Validate a Python dictionary against a registered JSON schema."""
    if schema_name not in SCHEMAS:
        return False, [f"Unknown schema name: '{schema_name}'"]
    schema = SCHEMAS[schema_name]
    try:
        jsonschema.validate(instance=data, schema=schema)
        return True, []
    except jsonschema.ValidationError as ve:
        return False, [f"Validation error at {ve.json_path}: {ve.message}"]
    except Exception as e:
        return False, [str(e)]

if __name__ == "__main__":
    saved = export_schemas()
    print(f"Exported {len(saved)} schemas to data/schemas/")
