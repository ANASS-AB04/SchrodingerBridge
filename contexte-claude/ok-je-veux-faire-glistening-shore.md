# Plan du rapport de stage — proposition de structure (v2)

## Contexte

Rapport de stage de 15 pages de **corps** (hors page de garde, résumé, sommaire,
bibliographie et annexes), à rédiger en français.

**Lectorat** : enseignants-chercheurs en **mathématiques** et en **mécanique**. Ils
suivront sans difficulté Fokker–Planck, un système d'optimalité, un schéma volumes
finis, la théorie du choc oblique. Ils ne sont **pas** spécialistes de recalage de
formes (*registration*, FFD) ni d'apprentissage. Trois notions doivent donc être
introduites explicitement plutôt que supposées connues :

1. ce qu'est une **registration FFD** et pourquoi elle produit un champ de déplacement ;
2. ce que signifie **conditionner un pont par une dérive de référence** β ;
3. le **recuit en γ** (ε-scaling) et pourquoi un IPFP démarré à froid à petit γ diverge.

**Choix arrêtés** :
- périmètre : le pont de Schrödinger est le cœur, le solveur Euler est présenté comme
  **l'outil développé** pour produire le corpus ;
- les résultats négatifs sont **discutés en fin de partie résultats** (§6.4), avant
  l'application — ordre conservé délibérément, le rapport ne finit pas sur les échecs
  mais ne les repousse pas non plus après la démonstration de rentabilité ;
- les 15 pages sont du corps seul ;
- **v2** : ajout d'un état de l'art dédié, du contexte de stage, et d'une sous-section
  de validation du solveur de pont.

**Matériau déjà rédigé, réutilisable tel quel** :
- [SB/referrence_drift.tex](SB/referrence_drift.tex) (470 l.) — dérivation complète :
  notation, problème de Schrödinger, Euler–Lagrange, Hopf–Cole, IPFP en log-potentiels,
  dérive de référence, équations transport–diffusion, cartes de transport, CDI, drift
  analytique du diamant (θ–β–M). Matière brute des parties 4 et 5.
- [SB/SB_results.pdf](SB/SB_results.pdf) — 31 slides, figures sélectionnées et
  légendées ; `outputs/deck/` contient les PNG prêts à inclure.
- [SB/SB.tex](SB/SB.tex) — validations 1D/2D synthétiques (gaussienne → bimodale) :
  alimente directement la nouvelle §4.5.
- **Absent** : aucune bibliographie (`.bib`), à créer.

---

## Plan proposé (15 p de corps)

| § | Titre | Pages |
|---|---|---|
| 1 | Introduction | 1,5 |
| 2 | État de l'art | 1 |
| 3 | Cadre physique et outil de simulation | 1,5 |
| 4 | Le pont de Schrödinger | 3 |
| 5 | Du pont abstrait au champ de Mach | 3 |
| 6 | Validation et résultats | 3,5 |
| 7 | Application : initialisation du solveur | 1 |
| 8 | Conclusion et perspectives | 0,5 |
| | **Total** | **15** |

### 1. Introduction · **1,5 p**

- **1.1 Cadre du stage** (~0,4 p) — **INRIA**, équipe et encadrement, durée, objectifs
  fixés au départ et comment ils ont évolué. Dire en une phrase que le travail s'inscrit
  dans un effort collectif (solveur JAX-FVM de G. de Romémont, cas tests de J. Léger),
  ce qui prépare l'attribution de §3.2 et évite d'y revenir en défense. Court et factuel.
- **1.2 Problématique** (~0,7 p) — études paramétriques en aérodynamique : chaque point
  de fonctionnement coûte une résolution complète des équations d'Euler. Les modèles
  réduits à base linéaire échouent sur les écoulements à chocs. **Formuler l'argument du
  *n*-width de Kolmogorov en une phrase intuitive** : interpoler linéairement deux
  discontinuités situées à des endroits différents en produit *deux*, atténuées, au lieu
  d'une seule déplacée. Figure d'appel en 1D (deux créneaux décalés + leur moyenne).
  D'où l'idée : traiter l'interpolation comme un **problème de transport de mesure**.
- **1.3 Contributions et organisation** (~0,4 p) — liste courte, et **délimitée** : le
  solveur Euler préexiste au stage (voir §3.2), la contribution est (i) le module de
  pont de Schrödinger et sa reconstruction, (ii) les trois dérives de référence dont
  l'auto-conditionnement SB², (iii) les nouvelles géométries et le corpus paramétrique,
  (iv) le protocole de validation à 586 runs, (v) l'application au démarrage à chaud.
  Annonce du plan.

### 2. État de l'art · **1 p**

Section neuve en v2. Trois paragraphes, un par famille :

- **Réduction de modèle pour écoulements à chocs** — pourquoi POD et bases linéaires
  butent ; les stratégies de contournement (recalage, changement de variables, méthodes
  adaptatives).
- **Approches par transport et par recalage** — Iollo & Taddei (transport optimal à
  modèles gaussiens pour structures cohérentes) et la CDI de Cucchiara *et al.* C'est
  ici qu'on **explique la notion de *mapping*** au lecteur non spécialiste, avant d'en
  avoir besoin au §5.3.
- **Résolution numérique des ponts de Schrödinger** — Chen–Georgiou–Pavon pour la
  théorie et la convergence, De Bortoli *et al.* et Tong *et al.* pour montrer que le
  sujet est actif. Une phrase chacun, pas plus : ce n'est pas le public.

Terminer par la phrase de positionnement : *le stage substitue aux cartes de la
registration les cartes issues d'un pont de Schrödinger, et mesure ce que cela change.*

### 3. Cadre physique et chaîne de simulation · **1,5 p**

Resserré de 2 p à 1,5 p en v2. **Section à attribution explicite** : le solveur n'est pas
un travail du stage, et il faut que le jury le lise sans ambiguïté dès la première
phrase de §3.2 — c'est la lecture honnête, et elle met d'autant mieux en valeur ce qui
est réellement produit ensuite.

- **3.1** Euler 2D compressible, régime transsonique/supersonique ; l'objet à interpoler
  est le champ de Mach $M(\mathbf{x};\,M_\infty,\alpha)$. Rappel de la structure d'onde
  attendue (choc oblique attaché / choc de proue détaché) — c'est ce que la méthode doit
  savoir déplacer.
- **3.2 Le solveur utilisé** (et non développé). Filiation à énoncer clairement :
  - la **base du solveur est JAX-FVM, de Guillaume de Romémont** — volumes finis
    différentiables et entropiquement stables sur maillages non structurés pour
    écoulements compressibles ; **à citer** (arXiv:2607.07385) ;
  - **Julien Léger**, membre de l'équipe, a construit sur ce code les cas tests
    initiaux (dièdre, bosse en canal) et le pilotage utilisés ici ;
  - le stage **reprend ces cas tests tels quels** et en ajoute de nouveaux **en
    respectant les conventions existantes** (notations, format de maillage, structure
    des sorties) ;
  - une phrase, pas plus, sur la participation au solveur : le stage y a contribué par
    des **retours et des tests**, ce dont l'article JAX-FVM fait état en remerciements
    (« *A deep thank you to Julien, Anass and Mathias for their support and feedback* »).
    **Formulation à surveiller** : un remerciement n'est pas une co-signature — écrire
    « contribution par des retours et des tests, mentionnée en remerciements », jamais
    une tournure qui laisserait entendre une co-paternité du solveur.
  Décrire en deux ou trois phrases seulement ce qu'il faut savoir pour lire la suite :
  flux HLLC, reconstruction MUSCL, intégration SRK2, conditions aux limites par
  marqueurs de faces, implémentation JAX (`lax.scan`, JIT) sur GPU.
- **3.3 Ce qui a été ajouté pour ce travail.** C'est ici que la contribution propre du
  stage côté CFD est énoncée, séparément :
  - **génération de maillages** pour de nouvelles géométries — profils NACA0012,
    RAE2822, OA209 — avec **raffinement de bord d'attaque** (`--nose-factor`) et
    maillage solveur imbriqué, de sorte que le choc de proue détaché soit résolu sans
    renchérir le coût côté SB
    ([Euler/generate_mesh.py](Euler/generate_mesh.py), [Euler/meshing/](Euler/meshing/)) ;
  - **production du corpus paramétrique** : grilles Mach × incidence sur SLURM, export
    des *bundles* `.npz` consommés par le module SB ;
  - la fonctionnalité de **démarrage à chaud** et son protocole de mesure
    ([Euler/warmstart.py](Euler/warmstart.py),
    [Euler/postproc/warmstart_analysis.py](Euler/postproc/warmstart_analysis.py)),
    qui alimente directement le §7.
- **3.4 Le corpus produit** : trois géométries retenues pour l'étude (dièdre, bosse en
  canal, NACA0012), maillages $h=0.025$, balayage Mach 0.80→3.00 et incidence 0°→4°.
  Donner le volume (nombre de solutions, coût GPU) pour matérialiser l'effort.
- Figures : un maillage + trois champs de Mach de référence (un par géométrie).

> **À confirmer avant rédaction** : les fichiers listés en §3.3 ne sont pas encore
> commités (ils apparaissent en `A`/`??` dans l'arbre de travail de la branche
> `euler-update`), donc l'historique git ne les attribue à personne. Vérifier qu'ils sont
> bien du stage avant de les revendiquer.

### 4. Le pont de Schrödinger · **3 p** — cœur mathématique

Source : [SB/referrence_drift.tex](SB/referrence_drift.tex) §1–3, à condenser.

- **4.1 Du transport optimal au problème de Schrödinger.** Formulation dynamique :
  chercher $(\rho_t,\mathbf{b}_t)$ vérifiant Fokker–Planck
  $\partial_t\rho+\nabla\!\cdot(\rho\mathbf{b})=\gamma\Delta\rho$ avec $\rho_0=\mu$,
  $\rho_1=\nu$, minimisant l'énergie cinétique espérée. Girsanov ⇒ équivalence avec une
  projection KL sur un processus de référence. **Dire pourquoi Schrödinger et pas
  Benamou–Brenier pur** : la régularisation entropique rend le problème strictement
  convexe et résoluble par point fixe, et $\gamma\to0$ redonne le transport optimal.
- **4.2 Système d'optimalité et linéarisation de Hopf–Cole** : le couple non linéaire
  devient deux équations de transport–diffusion découplées sur $(\eta,\eta^\ast)$.
- **4.3 IPFP en log-potentiels** : c'est Sinkhorn, mais **sans matrice noyau explicite**
  — le noyau est l'opérateur de semi-groupe appliqué par résolution d'EDP. Souligner ce
  point : c'est l'astuce qui rend la méthode viable sur maillage non structuré à
  $\sim2\times10^4$ cellules.
- **4.4 Mise en œuvre numérique** : discrétisation VF du semi-groupe (TPFA + SSP-RK2),
  accélération d'Anderson, **recuit en γ** (chaque étage réinitialisé par les potentiels
  du précédent), nécessité de la double précision.
- **4.5 Validation sur cas synthétiques** (~0,5 p, neuve en v2) — 1D square → OU, 2D
  gaussienne → bimodale sur maillage non structuré. Établit que **le solveur de pont est
  juste** avant de l'appliquer à la CFD, où l'on n'a plus de solution de référence
  analytique. Source : [SB/SB.tex](SB/SB.tex).

### 5. Du pont abstrait au champ de Mach · **3 p** — contribution méthodologique

Source : [SB/referrence_drift.tex](SB/referrence_drift.tex) §3.5–5.

- **5.1 Quelle mesure transporter ?** Un champ de Mach n'est pas une densité de
  probabilité. Construction de la marginale : métrique de Hessienne d'Alauzet–Loseille
  (mode `eig`, $|\lambda_{\max}|$), qui concentre la masse **sur les chocs**. Dire
  pourquoi le mode `det` a été écarté (masse nulle sur les flancs droits du diamant).
- **5.2 Revenir au champ physique.** Cartes barycentriques $T$, $S$ comme moyennes
  conditionnelles à travers le semi-groupe, correction du biais de fuite de Neumann,
  puis **interpolation par déplacement convexe (CDI)** :
  $\widehat M(t,\mathbf{x})=(1-t)M_0(\cdot)+tM_1(\cdot)$ évalué aux pré-images
  déplacées. Insister : **sans inversion de carte et sans gradient**.
- **5.3 La dérive de référence β.** Motivation : à petit γ le pont sans dérive ne
  converge plus. Conditionner le processus de référence pour qu'il sache déjà à peu près
  où va le choc. Trois variantes par généralité croissante :
  (i) **analytique** — rotation rigide fenêtrée déduite de θ–β–M (dièdre seulement) ;
  (ii) **FFD** — *expliquer ici la registration pour un non-spécialiste* : on plonge les
  deux marginales dans une grille de points de contrôle, on optimise leur déplacement
  pour superposer les deux images, le champ de déplacement obtenu fournit β ;
  (iii) **SB²** — auto-conditionnement : l'étage $k{+}1$ prend pour dérive la solution
  du pont de l'étage $k$ ; la solution exacte est un point fixe de cette itération.
- Figure : les deux marginales + le champ β + la séquence de transport de densité.

### 6. Validation et résultats · **3,5 p**

- **6.1 Protocole.** Bandes de Mach de largeur 0.10 ; chaque interpolation notée contre
  **9 solutions Euler de référence** à $t=0.1\ldots0.9$. Trois interpolateurs comparés :
  **BaryCDI** (le pont), **Linear** (moyenne pondérée, témoin naïf) et **FFD** (la
  registration utilisée *directement* comme interpolateur — témoin qui isole l'apport
  propre du pont). Métriques : $L^2$, $L^\infty$, $W_2$, $C_D$, $C_L$. Volume : 586 runs.
- **6.2 Qualitatif.** Deux bandes : bosse $M\,2.50\to2.60$ et NACA0012 $M\,1.80\to1.90$
  à 2°. Champ reconstruit vs. linéaire, puis cartes d'erreur absolue : le pont **place**
  le choc, la moyenne pondérée en produit deux atténués. C'est la figure qui porte le
  rapport.
- **6.3 Quantitatif.** Balayage $L^2$ et $L^\infty$ en fonction de la bande de Mach, avec
  les trois régimes physiques distingués. Gains : SB sous Linear sur **19 bandes sur 22**
  pour la bosse (gain médian 1,31×), jusqu'à 1,73× sur le profil sous $M\approx2$. Puis
  les coefficients aérodynamiques sur le même balayage.
- **6.4 Limites** (≈1 p) :
  - la **registration FFD reste devant** le pont en $L^2$ à tous les Mach, pour un coût
    hors-ligne ~100× moindre — question ouverte du stage ;
  - sur $C_D$ l'interpolation linéaire gagne d'un facteur 3 à 7 : la marginale ne porte
    aucune masse à la paroi, là où les efforts sont calculés ;
  - l'**axe d'incidence** est un échec net (SB 40× pire que Linear sur la seule bande
    NACA menée à terme) : 1° d'incidence change l'**amplitude** du système d'ondes plus
    qu'il ne le déplace — le transport est le mauvais *a priori* ;
  - au budget d'itérations fixé, l'IPFP n'atteint la tolérance que dans 6 à 10 runs sur
    22 selon le cas.

### 7. Application : initialisation du solveur · **1 p**

Resserré de 1,5 p en v2 : une figure + une table, prose minimale.

- Le débouché : un champ interpolé n'est pas seulement une réponse, c'est une
  **initialisation** pour le solveur à un Mach jamais calculé.
- Protocole : nombre de pas fixe, seuil de stationnarité désactivé, historique de
  convergence complet — « itérations pour atteindre la tolérance $X$ » se relit
  *a posteriori* pour tout $X$.
- Message méthodologique : le **facteur** d'accélération dépend entièrement de la
  tolérance d'arrêt (30× à 11× sur la plage atteignable) ; ce qui est stable, c'est le
  **nombre d'itérations économisées** (~3600). Citer un facteur sans sa tolérance n'a pas
  de sens.
- Rentabilité : contre un démarrage à froid, le pont est amorti en **6 requêtes**.

### 8. Conclusion et perspectives · **0,5 p**

Ce qui est établi, ce qui ne l'est pas, et deux ou trois pistes : marginale portant la
paroi pour récupérer $C_D$, budget IPFP, maillage raffiné au bord d'attaque.

---

## Annexes proposées

| # | Contenu | Source |
|---|---|---|
| A | Relation θ–β–M et construction du drift oblique | `referrence_drift.tex` §5 |
| B | Détail de la registration FFD | `SB/ffd_drift.py`, `phdtruel/` |
| C | Bootstrap SB² / SB²_exact (indexation temporelle du noyau, symétrie ⟨Qu,v⟩=⟨u,Q†v⟩) | `SB/utils.py` |
| D | Tableaux complets du balayage (22 bandes × 3 dérives × 3 γ, par cas) | `outputs/*/_sweep/sweep_summary_eig.csv` |
| E | Étude de convergence en maillage (échelle de $h$, diamant) | `outputs/Mach_interpolation/diamond/.../eig/h*/` |
| F | L'axe d'incidence : le cas d'échec en détail | `outputs/AoA_interpolation/naca0012/` |
| G | Vérification du solveur Euler + détails d'implémentation (JAX, chaîne SLURM, reproductibilité) | `CLAUDE.md`, `*.sbatch` |

---

## Placement de la bibliographie fournie

Aucun `.bib` n'existe encore : à créer (p. ex. `report/refs.bib`).
**Numérotation mise à jour pour la v2.**

### Références structurantes — à citer dans le corps

| Référence | Où | Rôle dans le rapport |
|---|---|---|
| **Cucchiara, Iollo, Taddei, Telib** — *Model order reduction by convex displacement interpolation* | **§5.2** (et **§2**) | **La** référence de la reconstruction CDI. La formule $\widehat M(t,\mathbf{x})$ implémentée dans [SB/utils.py](SB/utils.py) est leur CDI ; le stage en change la source des cartes (potentiels du pont au lieu de la registration). |
| **Iollo & Taddei** — *Mapping of coherent structures in parameterized flows by learning optimal transportation with Gaussian models* (JCP 2022) | **§2** puis **§5.3** | Travail antérieur le plus proche : écoulements paramétrés, structures cohérentes, transport optimal pour construire l'application. Justifie le positionnement du stage et introduit la notion de *mapping*. |
| **Chen, Georgiou, Pavon** — *Optimal transport over a linear dynamical system* (2015) | **§4.1** puis **§5.3** | Justification théorique de la **dérive de référence** : transporter au-dessus d'un processus a priori possédant sa propre dynamique. À placer au moment où β est introduite, sinon β passe pour un artifice numérique. |
| **Chen, Georgiou, Pavon** — *Entropic and displacement interpolation: a computational approach using the Hilbert metric* (2015) | **§4.3** et **§4.4** | Convergence de l'IPFP par contraction de Birkhoff. Double emploi : (i) le point fixe existe et est atteint ; (ii) le taux se dégrade quand $\gamma\to0$ — **c'est l'argument qui motive le recuit en γ**. *Displacement interpolation* fait aussi le pont vers §5.2. |
| **Caluya & Halder** — *Reflected Schrödinger Bridge: Density Control with Path Constraints* (2020) | **§4.4** et **§5.2** | Le pont à **bords réfléchissants** : justifie les conditions de Neumann sans flux du semi-groupe et le **flot déterministe réfléchi** $\Phi_\beta$ du biais de fuite. Référence spécifique et bien exploitée. |
| **Pavon, Tabak, Trigila** — *The data-driven Schrödinger bridge* (2018) | **§4.1** ou **§5.1** | Le pont lorsque les marginales viennent de **données** et non d'expressions analytiques — le cas ici. Cadre la construction de §5.1. |
| **de Romémont** — *JAX-FVM: A differentiable, entropy-stable finite volume solver on unstructured meshes for compressible flows* (arXiv:2607.07385) | **§3.2** | **Le solveur utilisé.** Citation obligatoire : c'est la base du code CFD, elle n'est pas un produit du stage. Fournit aussi la référence du schéma (VF entropiquement stable, maillages non structurés), ce qui rend une citation de Toro facultative plutôt que nécessaire. |

### Références de panorama — état de l'art et perspectives

| Référence | Où | Rôle |
|---|---|---|
| **De Bortoli, Thornton, Heng, Doucet** — *Diffusion Schrödinger Bridge* (2021) | **§2** (une phrase) et **§8** | L'IPFP est redevenu un outil central (modèles génératifs) et sa résolution numérique est un sujet actif. Situer, sans développer devant ce jury. |
| **Tong et al.** — *Simulation-Free Schrödinger Bridges via Score and Flow Matching* (2024) | **§8** | Voie qui **évite l'IPFP** : pertinente en perspective, puisque le coût hors-ligne de l'IPFP (~22 s à ~1240 s selon le cas) est précisément ce qui empêche le pont d'être rentable face à FFD. |
| **Zhu et al.** — *Neural Sinkhorn Gradient Flow* (2024) | **§8** | Même rôle : Sinkhorn appris plutôt qu'itéré. Périphérique, une ligne. |
| **Caissard, Coeurjolly, Lachaud, Roussillon** — *Heat kernel Laplace-Beltrami operator on digital surfaces* (2017) | **§4.4** | Appui sur la **discrétisation du noyau de la chaleur** comme objet numérique fiable. Attention : le papier traite de surfaces *digitales*, le lien avec le schéma VF/TPFA sur maillage triangulaire est **analogique** — le formuler comme tel. |

### Manques à combler

- **§5.1 — métrique hessienne** : la marginale repose sur la métrique d'**Alauzet &
  Loseille**. Manque le plus important : toute la construction de la mesure en dépend.
- **§5.3 — FFD** : la déformation de forme libre a besoin de sa source (Sederberg & Parry
  1986, ou la référence propre au framework `phdtruel/`).
- ~~**§3.2 — solveur**~~ : **comblé** par la référence JAX-FVM de de Romémont. Toro
  (*Riemann Solvers and Numerical Methods for Fluid Dynamics*) devient facultatif, à
  ajouter seulement si le détail de HLLC/MUSCL est développé.

Optionnels selon la place : Schrödinger (1932), Cuturi (2013) pour Sinkhorn entropique,
Benamou & Brenier (2000) pour la formulation dynamique, et une référence sur la barrière
du *n*-width en transport (Ohlberger & Rave) pour étayer le §1.2.

---

## Pages liminaires (hors des 15 p)

Le rapport sera rédigé dans [rapport/](rapport/), qui contient déjà les trois logos.

**Logos identifiés** (noms actuels à renommer — espaces et parenthèses dans un chemin
`\includegraphics` sont une source d'ennuis) :

| Fichier actuel | Logo | Nom proposé |
|---|---|---|
| `Image collée.png` | INRIA | `logos/inria.png` |
| `Image collée (2).png` | Groupe Ingeliance | `logos/ingeliance.png` |
| `Image collée (3).png` | Bordeaux INP – ENSEIRB-MATMECA | `logos/enseirb-matmeca.png` |

**Page de garde** — informations arrêtées :

| Champ | Valeur |
|---|---|
| Titre | **Résolution du problème du pont de Schrödinger pour la réduction de modèles non linéaires** |
| Auteur | Anass Aboufadel |
| Tuteur entreprise | **Mathias Truel** (seul) |
| Encadrant école | **Mathieu Colin** |
| Période | **26 mai – 18 septembre 2026** (≈ 16 semaines) |
| Formation | Bordeaux INP – ENSEIRB-MATMECA |
| Statut | **Stage Ingeliance, accueil dans l'équipe INRIA** — Ingeliance mis en avant comme employeur, INRIA comme structure d'accueil |

Disposition des logos : ENSEIRB-MATMECA en haut à gauche, Ingeliance en haut à droite
(employeur), INRIA en pied de page ou à droite sous Ingeliance.

*Casse du titre* : l'énoncé fourni était en capitales de titre à l'anglaise
(« Résolution Du Problème Du… »). En typographie française on ne capitalise que le
premier mot et les noms propres — d'où la forme retenue ci-dessus. À signaler si la
casse d'origine était volontaire.

Note : Mathias Truel est l'auteur du framework FFD vendorisé dans `phdtruel/`, qui
fournit la dérive `ffd` du §5.3 — le lien mérite d'être fait dans les remerciements.

**Résumé / Abstract** (~200 mots chacun, FR puis EN) — structure en cinq phrases :
problème (interpolation de champs à chocs, échec des bases linéaires) → approche (pont
de Schrödinger entropique, marginale hessienne, reconstruction CDI) → validation (586
runs, trois géométries) → résultat principal chiffré (gain sur $L^2$ contre
l'interpolation linéaire, accélération du solveur) → limite honnête (la registration
FFD reste devant). Mots-clés / keywords à la suite.

**Sommaire** — `\tableofcontents` en profondeur 2 (sections et sous-sections). Avec 8
sections et ~3 sous-sections chacune il tient sur une page ; ne pas descendre au niveau 3.

**Remerciements** — équipe INRIA, Mathias Truel, Mathieu Colin, et mention de Guillaume
de Romémont et Julien Léger pour le solveur et les cas tests repris.

## Points à préparer avant rédaction

- **Budget figures** : viser 10 à 12 figures pour 15 p. Les PNG sont dans
  `outputs/deck/` ; les figures de synthèse du balayage sont régénérables par
  [SB/deck_figs.py](SB/deck_figs.py).
- **Langue des figures** : les axes produits par [SB/plot.py](SB/plot.py) et
  [SB/deck_figs.py](SB/deck_figs.py) sont en anglais. Soit on l'assume (usage courant),
  soit on régénère — `deck_figs.py` est facile à traduire, les figures issues des runs
  demanderaient de modifier `plot.py` et de relancer les calculs.
- **Figure manquante** : l'illustration 1D du §1.2 (deux créneaux décalés et leur
  moyenne) n'existe pas, elle est à produire — c'est quelques lignes de matplotlib.

## Première étape de rédaction (demandée)

Produire dans [rapport/](rapport/) :

1. **`rapport/logos/`** — copies des trois PNG sous des noms propres (`inria.png`,
   `ingeliance.png`, `enseirb-matmeca.png`). Les originaux restent en place.
2. **`rapport/refs.bib`** — les dix références fournies + JAX-FVM (de Romémont), avec
   les clés utilisées dans le corps. Les trois manques (Alauzet–Loseille, FFD,
   éventuellement Toro) sont laissés en entrées `TODO` commentées, visibles à la
   compilation.
3. **`rapport/rapport.tex`** — document en français, classe `report`, avec :
   - la **page de garde** conforme au tableau ci-dessus, trois logos placés ;
   - les **remerciements** ;
   - le **résumé** et l'**abstract** (~200 mots chacun, structure en cinq phrases,
     mots-clés / keywords) ;
   - le **sommaire** (`\tableofcontents`, profondeur 2) ;
   - le **§1 Introduction** entièrement rédigé (1.1 cadre du stage, 1.2 problématique,
     1.3 contributions et organisation), ~1,5 p.
4. Vérifier que le document **compile** (`pdflatex` ×2 + `bibtex`) et annoncer le nombre
   de pages obtenu.

### Méthode de rédaction : une partie à la fois

Le plan des 8 parties ci-dessus reste la feuille de route complète et ne change pas.
La rédaction se fait **partie par partie**, chacune avec sa passe dédiée, pour obtenir
la meilleure qualité d'écriture sur chacune plutôt qu'un premier jet global.

**Passe en cours — pages liminaires + partie 1 uniquement.** Les §2 à §8 ne sont pas
encore écrits, pas même en squelette : le sommaire ne listera donc que l'introduction
pour l'instant et se remplira à chaque passe suivante. La figure 1D du §1.2 n'est pas
produite : un emplacement réservé est laissé dans le texte.

**Passes suivantes** (à lancer une par une, dans cet ordre) : §2 état de l'art → §3
chaîne de simulation → §4 le pont → §5 du pont au champ de Mach → §6 validation et
résultats → §7 démarrage à chaud → §8 conclusion. Chaque passe ajoute sa partie au même
`rapport.tex` et recompile.

## Vérification

- Compter les pages du corps une fois la trame LaTeX posée avec les figures réelles : le
  budget est serré, **§6 est la première partie à déborder**, puis §5.
- Relire §4 et §5 en se demandant, à chaque paragraphe, si un mécanicien non
  mathématicien et un mathématicien non mécanicien s'y retrouvent tous les deux.
- Vérifier que chaque chiffre cité au §6 est retrouvable dans un `sweep_summary_eig.csv`
  ou un `metrics.json`, et pas seulement dans les slides.
