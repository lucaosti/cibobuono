# CiboBuono — Site (React + TypeScript + Vite)

Interactive map of Italian food venues reviewed on YouTube. Deployed automatically to GitHub Pages via GitHub Actions on every push to `main`.

## Stack

| Component | Tool |
|-----------|------|
| Framework | React 18 + TypeScript |
| Build | Vite |
| Map | Leaflet + react-leaflet |
| Tiles | OpenStreetMap |
| i18n | Custom context (EN / IT, `localStorage`) |
| Data | `data/*.json` fetched at runtime via relative URLs |

## Local Development

```bash
# from repo root
cd site
npm install
npm run dev      # http://localhost:5173 (hot-reload)
npm run build    # production build → site/dist/
npm run preview  # preview built output locally
```

The dev server proxies `data/` from the repo root so the map works without a full pipeline run.

## Structure

```
site/
├── index.html              # HTML entry — title + meta description
├── vite.config.ts          # base: /cibobuono/ (GitHub Pages path)
├── package.json
└── src/
    ├── App.tsx             # Data loading, filtering, layout
    ├── api.ts              # Fetch JSON from data/ at runtime
    ├── types.ts            # TypeScript types (mirrors Python schemas)
    ├── i18n/
    │   ├── messages.ts         # Typed EN + IT message dictionaries
    │   ├── LanguageContext.tsx  # Provider (persists language in localStorage)
    │   └── useLanguage.ts      # useLanguage / useT hooks
    └── components/
        ├── MapView.tsx         # Leaflet map, sentiment-coloured markers, popups
        ├── LocaleList.tsx      # Sidebar list, expandable visit details
        ├── Header.tsx          # Search + category/sentiment filters + EN/IT toggle
        └── StatusBar.tsx       # Loading / error states
```

## Deployment

GitHub Actions (`.github/workflows/deploy.yml`) runs `npm run build` and copies `data/` into `site/dist/` on every push to `main`. The built site is then deployed to GitHub Pages.

Enable Pages: **Settings → Pages → Source: GitHub Actions**.

## Data

The site reads these files at runtime from `data/` (relative to the app root):

| File | Content |
|------|---------|
| `channels.json` | YouTube channels |
| `videos.json` | All cataloged videos |
| `locales.json` | Normalised venues with coordinates |
| `visits.json` | Visit/review entries |

No data is bundled into the JavaScript — `data/` is copied alongside the build output at deploy time, so the dataset stays in version control and is easily diffable.

## Bilingual UI

Language is auto-detected from `navigator.language` (Italian first, English fallback). User choice persists in `localStorage`. The EN/IT toggle in the header switches on the fly without a page reload. All UI strings live in `src/i18n/messages.ts`.
