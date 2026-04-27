# Slide presentazione baseline

`slides.md` è una presentazione [Marp](https://marp.app/) editabile come puro markdown.

## Editing

Apri `slides.md` in qualunque editor. Per la **preview live** in VSCode:

1. installa l'estensione **Marp for VS Code**
2. apri `slides.md`
3. clicca l'icona "Open Preview to the Side" in alto a destra → vedi le slide aggiornarsi mentre scrivi

## Export

Dalla command palette di VSCode (con l'estensione Marp installata):

- `Marp: Export slide deck` → scegli **PDF**, **PPTX**, **HTML**, **PNG**

Da CLI (se installi `@marp-team/marp-cli` con `npm install -g @marp-team/marp-cli`):

```bash
marp slides.md --pdf
marp slides.md --pptx
marp slides.md --html
```

## Modificare i grafici

I PNG in `img/` sono stati generati dai PDF in `../results/` con `sips`. Se aggiorni un PDF, rigenera il PNG:

```bash
sips -s format png --resampleHeightWidthMax 1600 ../results/NOME.pdf --out img/NOME.png
```

## Struttura del file

Ogni slide è separata da `---`. Il blocco di stile in cima (`<style>`) controlla colori, dimensioni font e i 3 box colorati:

- `<div class="takeaway">` — giallo, per i punti chiave
- `<div class="caveat">` — rosso, per i caveat metodologici
- `<div class="ok">` — verde, per le note positive
