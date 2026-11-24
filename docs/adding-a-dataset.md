# Recipe: add a new dataset format

Dataset loading is **backend-only** and nicely decoupled — a new *format* is one
loader function plus one line of dispatch. (Adding a new *kind* of data beyond
tensors/images is more involved; see the note at the end.)

All of this is in `server/src/tensorcanvas_server/datasets.py`.

## 1. Write a loader

A loader takes the raw bytes and returns a `Dataset`. Raise `DatasetError` with a
clear, user-facing message on anything malformed — those messages surface
directly in the UI.

```python
def load_tsv(data: bytes, name: str) -> Dataset:
    text = data.decode("utf-8", errors="replace")
    rows = [line.split("\t") for line in text.splitlines() if line.strip()]
    if not rows:
        raise DatasetError("TSV is empty")
    # ... parse features + target, build arrays ...
    return Dataset(id=..., name=name, X=X, y=y, ...)   # match the existing loaders
```

Look at `load_csv` for the exact `Dataset` fields and conventions (feature matrix
`X`, targets `y`, optional class labels).

## 2. Register it in the dispatch

Add a branch in `load_upload()` for the file extension:

```python
def load_upload(filename: str, data: bytes) -> Dataset:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".csv":
        return load_csv(data, filename)
    if ext == ".tsv":                      # ← new
        return load_tsv(data, filename)
    ...
    raise DatasetError(f"unsupported file type '{ext}'; use .csv, .tsv, .json, ...")
```

Update the `unsupported file type` message so the accepted list stays accurate.

## 3. Let the UI accept the extension

The upload inputs filter by extension. In the client, widen the `accept` list
(e.g. `components/observe/InputPicker.tsx` and the train upload control) to
include `.tsv`.

## 4. Verify

```bash
cd server && uv run pytest
```

Add a test that feeds a small in-memory file of your format to `load_upload` and
checks the resulting `Dataset`. Then try a real file through the UI.

---

### Adding a new *kind* of data (not just a format)

If the data isn't a table of features/target or class-foldered images — say,
sequences or graphs — you also have to teach `match_to_network()` how to coerce
it to the model's input/output shape, and likely `preview()` and the observe
input resolution. That's a larger change; open an issue to discuss the shape of
it first.
