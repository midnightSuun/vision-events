"""Write Pydantic models from the migrated Postgres schema.

The database is the source of the shape. Run this after dbmate applies a migration:

    uv run python scripts/generate_models.py
"""

from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path

import psycopg

_TYPE_BY_UDT = {
    "int2": "int",
    "int4": "int",
    "int8": "int",
    "text": "str",
    "varchar": "str",
    "bpchar": "str",
    "name": "str",
    "bool": "bool",
    "json": "dict[str, Any]",
    "jsonb": "dict[str, Any]",
    "timestamptz": "datetime",
    "timestamp": "datetime",
    "float4": "float",
    "float8": "float",
    "numeric": "float",
    "uuid": "str",
}

_COLUMNS = """
SELECT table_name, column_name, udt_name, is_nullable
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name <> 'schema_migrations'
ORDER BY table_name, ordinal_position
"""


def _class_name(table_name: str) -> str:
    return "".join(part[:1].upper() + part[1:] for part in table_name.split("_") if part)


def _annotation(udt_name: str, nullable: str) -> str:
    base = _TYPE_BY_UDT.get(udt_name, "Any")
    if nullable == "YES":
        return f"{base} | None"
    return base


def _render(tables: dict[str, list[tuple[str, str]]]) -> str:
    needs_any = False
    needs_datetime = False
    body: list[str] = []
    for table_name, columns in tables.items():
        body.append(f"class {_class_name(table_name)}(BaseModel):")
        if not columns:
            body.append("    pass")
            body.append("")
            continue
        for column_name, annotation in columns:
            body.append(f"    {column_name}: {annotation}")
            if "Any" in annotation:
                needs_any = True
            if "datetime" in annotation:
                needs_datetime = True
        body.append("")

    imports = ["from pydantic import BaseModel", "", ""]
    if needs_datetime or needs_any:
        typing_lines: list[str] = []
        if needs_datetime:
            typing_lines.append("from datetime import datetime")
        if needs_any:
            typing_lines.append("from typing import Any")
        imports = typing_lines + ["", *imports]

    header = [
        '"""Generated from the live Postgres schema. Do not edit by hand.',
        "",
        "Regenerate after a migration:",
        "",
        "    uv run python scripts/generate_models.py",
        '"""',
        "",
    ]
    return "\n".join(header + imports + body).rstrip() + "\n"


def main() -> None:
    database_url = os.environ.get(
        "DATABASE_URL",
        "postgresql://vision:vision@localhost:5432/vision",
    )
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(_COLUMNS)
            rows = cur.fetchall()

    tables: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for table_name, column_name, udt_name, nullable in rows:
        tables[str(table_name)].append(
            (str(column_name), _annotation(str(udt_name), str(nullable)))
        )

    target = Path(__file__).resolve().parents[1] / "app" / "db_models.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_render(dict(tables)), encoding="utf-8")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
