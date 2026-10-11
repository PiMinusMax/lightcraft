# Joint-NR: Plan zur Beschleunigung ohne Qualitätsabbau

Stand: 2026-10-11, geprüft gegen `8b89526d` auf `nr-color-v2`.
Dies ist ein Umsetzungsplan; der Filter wurde dabei nicht verändert.

## Entscheidung

Zuerst dieselbe Joint-Berechnung effizienter ausführen. Rauschmodell, Pilot,
gemeinsame Farbgewichte, negative Werte, symmetrischen Austausch und Schutz vor
dem Überspringen dünner Kanten unverändert lassen. Eine Halbierung der Auflösung,
weniger Stufen oder eine Rückkehr zum alten Guided Filter wären Änderungen des
Algorithmus und haben derzeit keinen Nachweis gleichbleibender Qualität.

Ein deutlich schnellerer CPU-Pfad ist plausibel. Dass er das Budget von **1 s für
den gesamten 24-MP-JPEG-Export** erreicht, ist noch nicht belegt. Das ist kein
1-s-Budget allein für NR. Ein verbindliches zusätzliches RAM-Limit ist in den
gelesenen Projektanweisungen nicht beziffert; deshalb Peak-RSS und Filterpuffer
separat ausweisen und vor einer Produktionsfreigabe ein Limit festlegen.

## Befund im aktuellen Code

Bezug: `crates/pipeline/src/nrc_joint.rs`.

| Stelle | Befund | Konsequenz |
| --- | --- | --- |
| `denoise_grid`, Zeilen 250–319 | Pro Stufe: Gewichte, vier Diffusionsdurchläufe, Pfad-Maxima | Bei NR 100 und Standardparametern sieben Stufen, 42 vollständige Durchläufe allein in dieser Schleife |
| `diffuse`, Zeilen 178–198 | Kopiert drei Kanäle; nur zwei werden verändert | Y wird ohne Nutzen durch jeden Diffusionspuffer transportiert |
| Zeilen 250–253 | `filtered`, `scratch`, `paths`, `links` sind alle RGB-Puffer | Die letzten beiden benötigen nur zwei Richtungen, keine dritte Komponente |
| Zeilen 254–268 und 279–289 | Distanz direkter Nachbarn wird bei delta=1 erneut berechnet | Bereits vorhandene Pfadwerte können die erste Gewichtsberechnung speisen |
| Zeilen 298–317 | Pfad-Maxima werden auch nach der letzten Stufe aufgebaut | Ein kompletter Durchlauf erzeugt ungenutzte Daten |
| Zeilen 298–317, danach 279–289 | Pfadbildung und Gewichtsbildung sind getrennte Durchläufe | Für die nächste Stufe in einem Durchlauf berechenbar |
| Zeilen 324–328 | RGB-Rekonstruktion auf voller Auflösung läuft seriell | Zeilenparallel ausführen; später mit letzter Diffusion verbinden |
| Zeilen 127–137, 279–289 | Drei Varianznormalisierungen pro Distanz, `exp` pro gültigem Paar | Rechenkosten neben Speicherverkehr; genaueres Profil nötig |
| `crates/gpu/src/render.rs`, Zeilen 786–790 | Download → CPU-NR → Upload | GPU-Betrieb beschleunigt Joint derzeit nicht |

Bei 32,7 MP und sieben Stufen: ungefähr 458 Millionen Paargewichte/Exponential-
auswertungen und 917 Millionen Diffusions-Pixelupdates, etwas weniger an Rändern.
Die sechs dauerhaften RGB-Arbeitsbilder belegen zusammen etwa 2,36 GB zusätzlich
zum Eingabebild und zur übrigen Pipeline. Die Stufen vervielfachen die Arbeit,
nicht die Anzahl gleichzeitig gehaltener Bilder.

### Kontrollmessung

Vorhandene `.target-nr/release/lightcraft-cli` vom 11.10., Katze 7008×4672,
`NRV2_JOINT=1`, `LIGHTCRAFT_GPU=0`, `LIGHTCRAFT_PROFILE=1`, NR 100,
Standard-Detail/Smoothness. Zwei einzelne Läufe auf i7-7820HQ; kein statistischer
Benchmark und kein Nachbau der gemeldeten 6,2 s unter identischen Bedingungen.

| Teil | 4 Rayon-Threads | 8 Rayon-Threads |
| --- | ---: | ---: |
| Transformation | 323 ms | 278 ms |
| Rauschmodell | 60 ms | 60 ms |
| Pilot | 837 ms | 633 ms |
| Stufe 1, einschließlich Pufferanlage und initialer Pfadbildung | 1761 ms | 1564 ms |
| Stufen 2–64 zusammen | 7816 ms | 5641 ms |
| Joint gesamt | 11029 ms | 8411 ms |
| Ganzer CLI-Lauf einschließlich RAW und PNG | 23,33 s | 18,70 s |
| Peak-RSS des Prozesses | 3,32 GiB | 3,32 GiB |

Logs: `/tmp/nrc-plan-cat100.log`, `/tmp/nrc-plan-cat100-t8.log`.
Die vorhandenen Stufentimer trennen Gewichte/Diffusion/Pfade noch nicht auf.
Eine alleinige Optimierung des Rauschmodells kann bei diesem Profil weniger als
1 % der Filterzeit sparen. PNG-Laufzeit ist nicht mit dem JPEG-Budget vergleichbar.

## Reihenfolge der Umsetzung

### 0. Referenz und belastbare Messung

- Gleichen Release-Build, feste CPU-/Threadkonfiguration und unveränderte
  Eingaben verwenden. Warm-up und mindestens fünf Wiederholungen; Median und
  Streuung, Peak-RSS, NR-Zeit und gesamten JPEG-Export getrennt aufzeichnen.
- Gewichtsbildung, H/V-Diffusion, Pfadbildung, Allokation und Rekonstruktion einzeln
  instrumentieren. Auch `release` gegen vorhandenes `dist` mit Thin-LTO prüfen;
  keinen Sonderbuild mit lokalen CPU-Instruktionen als allgemein gültigen Gewinn verkaufen.
- Float-Eingabe unmittelbar vor NR und Float-Ausgabe der Referenz sichern. Drei
  Kameramotive bei 0/25/50/100, zusätzlich 24 MP, Vorschau und Zoomfenster.
  Jede Optimierung einzeln gegen diese Referenz prüfen.

### 1. Überflüssige Arbeit entfernen, gleicher Algorithmus

1. `step == 1` direkt transformieren, ohne die generischen Resampling-Schleifen.
   Rechenreihenfolge zunächst beibehalten.
2. Bei delta=1 `exp(-path)` verwenden: für gültige Paare ist `path` bereits genau
   dieselbe Distanz. Keine zusätzliche Distanzberechnung.
3. Nach der letzten Stufe keine weiteren Pfad-Maxima berechnen.
4. Für Folgestufen Pfad-Maximum `P(2d)` und zugehöriges Paargewicht zusammen
   berechnen. Alten Pfadpuffer bis zum Ende des Durchlaufs nur lesen; Ergebnis in
   separaten Scratch schreiben. **Nicht in-place**, sonst werden Pfade verfälscht.
5. Rekonstruktion mit `for_rows` parallelisieren.

Bei sieben Stufen kann die Zusammenführung einschließlich Wegfall des letzten
Pfadaufbaus sieben der bisher 42 Schleifendurchläufe entfernen. Das sind rund
17 % weniger Durchläufe, **keine zugesagte Laufzeitersparnis von 17 %**: Die
verbleibenden Durchläufe sind unterschiedlich teuer.

Abnahme: bei unveränderter Gleitkomma-Reihenfolge bitgleiche Float-Ergebnisse;
Abweichungen untersuchen, nicht automatisch die Toleranz erhöhen.

### 2. Speicherformat und innere Schleifen

- Diffusion nur mit `[f32; 2]` für R−Y/B−Y; Pfade und Gewichte ebenfalls mit
  zwei Komponenten. Guide behält alle drei Komponenten.
- Im Vollauflösungsmodus `original` nach Pilot und Initialisierung des
  Chromapuffers freigeben. Die spätere Rekonstruktion liest Original-Y aus `img`.
  Den experimentellen `step=2`-Pfad getrennt halten: er benötigt `original` länger.
- Damit sinken die dauerhaften Arbeitsdaten rechnerisch von 72 auf 44 Byte/Pixel:
  bei 32,7 MP von 2,36 auf 1,44 GB, etwa **39 % weniger**. Bei 24 MP bleiben noch
  1,06 GB; das ist ein Zwischenschritt, keine ausreichende RAM-Lösung für sich.
- H/V-Kernel getrennt schreiben, Randbereiche getrennt behandeln und innen mit
  geprüften Zeilenslices und `zip` arbeiten. Ziel: weniger Indexarithmetik und
  bessere automatische Vektorisierung in sicherem Rust. Kein `unsafe`, keine
  pauschale Abschaltung von Prüfungen, kein Fast-Math.
- Originale Nachbar- und Additionsreihenfolge erhalten. Letzte vertikale Diffusion
  optional direkt in die RGB-Rekonstruktion schreiben, wenn danach kein gefilterter
  Zwischenstand mehr benötigt wird.

Abnahme: gleiche Float-Ausgabe für reine Layout-/Schleifenänderungen; isolierte
Zeit- und RSS-Messung. Formatänderung reduziert Bytes, garantiert aber keinen
bestimmten CPU-Speedup.

### 3. Cache und begrenzter Arbeitsbereich

Nach dem neuen Profil zwei Varianten getrennt bewerten:

- **Cache-Blöcke pro Stufe:** H/V/H/V in lokalen Puffern mit exakten Halos
  ausführen. Pro Achse wächst die Abhängigkeit dieser vier Schritte um `2*delta`.
  Nur gültige Kernbereiche zurückschreiben; Reihenfolge und Paargewichte erhalten.
  Dies reduziert Zwischenverkehr, kann durch doppelte Halo-Arbeit aber langsamer
  werden. Kleine deltas zuerst; breite Stufen nicht blind in winzige Kacheln zwingen.
- **Begrenztes RAM für große Exporte:** ganze Filterkette auf größeren Kacheln
  mit festem globalem Rauschmodell, unveränderter Eingabe und separater Ausgabe.
  Dafür beträgt allein der Diffusionshalo pro Achse `2*sum(delta)`, bei sieben
  Stufen 254 Pixel, plus Pilotstütze. Pfad-/Guide-Abhängigkeiten zusätzlich sauber
  herleiten. Kachelgröße und Anzahl paralleler Kacheln aus dem Speicherbudget
  bestimmen. Das ist primär eine RAM-Maßnahme, kein garantierter Tempogewinn.

Den Pilot nicht unbemerkt mit anderer Randbehandlung oder einem neuen lokalen
Rauschfit ersetzen. Neu gestartete gleitende Summen können Rundung ändern;
das muss gegen die Referenz geprüft werden. Ein bestandener Zoomtest beweist
noch keine Gleichheit für alle Kachelgrößen und alle Vorschaupfade.

### 4. GPU ohne vollständigen CPU-Rundweg

Wenn CPU nach Schritten 1–3 das Budget weiterhin verfehlt, dieselben kompakten
Operationen als GPU-Kernel umsetzen. Bild und Arbeitsdaten auf der GPU halten;
für den zunächst weiter auf CPU laufenden Rauschfit nur die fest definierten
Stichproben/Residualdaten übertragen, anschließend die wenigen Koeffizienten zurück.
Den vollständigen `read_rgb`/Upload-Rundweg entfernen. Synchronisation mitmessen.

F32 beibehalten; gemeinsame symmetrische Gewichte, Reihenfolge und Kantenpfade
erhalten. CPU bleibt Qualitätsreferenz und Fallback. GPU-Ergebnisse brauchen
numerische und visuelle Abnahme, da `exp`/FMA geräteabhängig abweichen können.
VRAM-Budget und kleinere integrierte GPUs berücksichtigen. Ein GPU-Port beseitigt
die Pufferkosten nicht automatisch; deshalb erst das kompakte Format entwickeln.

## Qualitäts- und Freigabekriterien

- Vorhandene 147 Pipeline-Tests und die ignorierten Farb-/Mittelwertproben grün;
  zusätzlich Referenzvergleich der optimierten Kernel, ungerade/kleine Dimensionen,
  Bildränder, Kachelgrenzen und wechselnde Threadzahlen.
- Signed-Float-Mittelwert und pixelweise Luminanz erhalten; keine neuen Clips.
  Zwei-Pixel-Linien, gleichhelle Farbkanten und Blatt neben Schwarz weiter schützen.
- Alle sechs Vergleichsraster neu prüfen. Nicht nur mittlere Sättigung: mittleres
  a*/b*, Farbton, Streuung/Flecken, lokale Farbsäume und weiße Einzelpunkte messen.
  Für reine CPU-Umbauten gilt der stärkere Anspruch gleicher Float-Pixel.
- Vorschau, Zoom und Export einschließlich höherer Ausgabetiefe prüfen. Globales
  Rauschmodell darf sich bei Kachelung nicht pro Kachel verändern.
- Budgetfreigabe erst nach Messung des ganzen Exports auf festgelegter Hardware.
  Ist schon Export ohne NR über Budget, diesen Befund separat ausweisen.
- Vor Commit reguläres `cargo xtask ci`; keine Tests abschwächen, um einen
  Performance-Umbau passend zu machen.

## Reglerwirkung als eigener Arbeitsschritt

`denoise()` begrenzt `(4 * amount)` bereits bei Color=25 auf 1. Danach ändern sich
nur Radius und letzte Stufen. Guide und Kantengewichte bleiben fest; lange Pfade
werden durch das größte Hindernis begrenzt. Das erklärt die geringe zusätzliche
Wirkung plausibel, beweist aber noch nicht den Anteil jeder Ursache.

Zunächst den bestehenden Stand beschleunigen und als Referenz behalten. Danach
eine eigene Änderung für die Stärkezuordnung entwickeln und bewerten. Mehr
Diffusionsschritte oder weichere Kantenbarrieren wären Qualitätsänderungen mit
neuem Laufzeit-/Farbsaumrisiko. Stufen wegen kleiner globaler Bilddifferenzen zu
streichen wäre ebenfalls kein nachgewiesen verlustfreier Performance-Fix.

Optional für interaktive Regler: bei unverändertem Eingang/Detail den festen
Guide und einen gemeinsamen Stufen-Zwischenstand innerhalb eines begrenzten
Caches wiederverwenden. Exakte Invalidierung bei Geometrie, Weißabgleich und allen
vorherigen Verarbeitungsschritten erforderlich. Das beschleunigt wiederholte
Bedienung, nicht den ersten Export; nicht sämtliche Vollbilder aller Stufen cachen.
