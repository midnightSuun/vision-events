# detections-service

`app/models.py` is generated from the live Postgres schema. Do not edit it by hand.

After a dbmate migration is applied, regenerate it from this directory:

```bash
uv run python -m sqlacodegen "postgresql+psycopg2://vision:vision@localhost:5432/vision" --outfile app/models.py
```

The command is sqlacodegen with the psycopg2 driver. The file must keep the generated-file notice at the top.
