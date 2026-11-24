## What & why

<!-- What does this change do, and why? Link any related issue (e.g. Closes #12). -->

## Type of change

- [ ] Bug fix
- [ ] New feature / module (layer / dataset / pretrained model / example)
- [ ] Refactor / chore
- [ ] Docs

## Checklist

- [ ] Backend tests pass — `cd server && uv run pytest`
- [ ] Frontend lints and builds — `cd client && yarn lint && yarn build`
- [ ] I updated docs if I changed an extension point or behaviour
- [ ] **If I added a layer type**, I updated **both** `server/.../builder.py`
      (`LAYERS`) **and** `client/components/build/nodeCatalog.ts`
      (`CATALOG` + `PALETTE_ITEMS`)
- [ ] **If I added a pretrained model**, tests stay offline (no weight downloads)
      and I added a matching example network

## Screenshots / notes

<!-- For UI changes, add before/after screenshots or a short GIF. -->
