# Mapping cognition onto brain networks: a first literature review

**Project: neuropsychologygraphs — review v1**
Scope: work that relates cognition, and its subdomains, to brain connectivity representations — most commonly structural (diffusion MRI tractography) or functional (fMRI correlation) network matrices, plus the lesion, meta-analytic, and machine-learning variants of the same question.

Corpus: 672 PubMed records retrieved 2026-10-22 via 23 thematic queries plus 53 targeted anchor lookups ([neuropsychologygraphs_corpus.csv]({{artifact:3a77460a-02bd-4f55-a234-5eb39d498ca4}})). Citations below are to records verified in that corpus; DOIs are reproduced from PubMed metadata (data source: PubMed, NCBI).

---

## 1. The question and why it is hard

The field asks a deceptively simple question: given a matrix **A** whose entries describe connection strength between brain regions, what in **A** corresponds to a person's cognitive ability — their general reasoning capacity, their working memory span, their vigilance, their vocabulary? Three framings coexist and are often conflated:

- **Localization of function to a subnetwork.** Which edges or communities implement attention, language, episodic memory? This is the classical mapping agenda transposed from regions to edges, descending from hierarchical accounts of cortical organization (Mesulam, 1998) and the multiple-demand account of domain-general control (Duncan, 2010).
- **Individual-differences prediction.** Does the connectome of a given person forecast *that person's* score? This is a statistical-learning question, not a localization question, and it is now the dominant idiom (Shen et al., 2017).
- **Causal dependence.** Does damaging or stimulating a connection change cognition? This is where lesion and stimulation work enters (Boes et al., 2015).

Answers to the three do not automatically transfer. A network that predicts fluid intelligence out-of-sample need not contain the regions whose damage impairs reasoning, and the most reliable edges are not necessarily the most behaviorally informative. Much of the literature's apparent inconsistency dissolves once the three framings are separated.

Volume has grown steeply: a fixed PubMed query combining connectivity terms with cognition and MRI returns 24 records for 2005 and 1,197 for 2025, roughly a 50-fold increase.

![Annual PubMed records matching a fixed connectivity-and-cognition query, 2005–2025. Counts are PubMed `total_count` values for the identical query restricted to each publication year, so they index indexing volume rather than any single construct.]({{artifact:190362ab-f271-4e00-91a3-7fbb75e98e70}})

---

## 2. How the network is represented

Every result in this literature is conditional on four representational choices, and the corpus shows each of them to be consequential rather than cosmetic.

**Nodes.** Parcellation determines what a "region" is. The common choices — intrinsic-connectivity networks (Yeo et al., 2011), multimodal areal parcels (Glasser et al., 2016), gradient-weighted local-global parcels at multiple resolutions (Schaefer et al., 2018), and subcortical gradients-derived atlases (Tian et al., 2020) — are not interchangeable, and the methodological literature on parcellation choice is substantial (Eickhoff et al., 2018). Node definition at the individual level matters measurably for behavior prediction: individual-specific network topography predicts cognition, personality and emotion (Kong et al., 2019), individual areal-level parcellations estimated from ten minutes of data generalize better than group approaches using 150 minutes and yield the best behavioral prediction (Kong et al., 2021), and functional topography of association networks varies substantially across individuals in youth (Cui et al., 2020). Precision-mapping work established the empirical basis for this: dense within-individual sampling reveals stable idiosyncratic topographies (Gordon et al., 2017), and variance decomposition across subjects, sessions and tasks shows that functional networks are dominated by common organizational principles and stable individual features, with much smaller task-state and day-to-day contributions (Gratton et al., 2018). Probabilistic precision atlases now package this for cohort use (Hermosillo et al., 2024).

**Edges.** Functional edges are typically Pearson correlations between regional time series; structural edges are streamline counts or microstructure-weighted tractography measures. The two are related but not redundant, and the mapping between them — structure–function coupling — is itself a cognitively relevant quantity (Suárez et al., 2020), with coupling in rostrolateral prefrontal cortex associated with executive performance and partially mediating age-related improvement in executive function across 727 youths (Baum et al., 2019). Communication-model edges (search information, diffusion efficiency, path transitivity) are an intermediate representation motivated by the observation that correlation is a poor proxy for signalling (Avena-Koenigsberger et al., 2017).

**Derived features.** Graph-theoretic summaries — efficiency, modularity, rich club, hubness — remain the most common reduction (Bullmore & Sporns, 2009; reviewed systematically by Farahani et al., 2019), with scale-dependence an acknowledged problem (Betzel & Bassett, 2016). Two alternative reductions have gained ground: low-dimensional connectivity **gradients** or manifolds, following the demonstration that the default mode network sits at the terminus of a principal sensory-to-transmodal gradient (Margulies et al., 2016) with a topographic account of DMN function built on it (Smallwood et al., 2021); and **whole-matrix multivariate features** fed directly to a predictive model, which bypass graph summaries entirely. Newer statistical machinery aims to test multi-subject connectome differences across topological scales without relying on heuristic graph statistics (Gopalakrishnan et al., 2025).

**State.** The matrix depends on what the participant was doing. Resting state is the default, but task-evoked connectivity amplifies behaviorally relevant individual differences (Greene et al., 2018); task-based FC outperforms rest for several cognitive phenotypes (Zhao et al., 2023, *NeuroImage*), with the advantage attributable in part to task-design-induced changes in activity rather than connectivity alone; naturalistic movie viewing outperforms rest, with usable predictions from under three minutes of a single clip and socially rich clips performing best (Finn & Bandettini, 2021); and "general functional connectivity" pooling rest and task yields more reliable and heritable individual differences than either alone (Elliott et al., 2019). State and network specificity interact, so prediction accuracy is not a property of a connectome but of a connectome-state-target triple (Kraljević et al., 2024).

---

## 3. Five families of mapping strategy

### 3.1 Subnetwork localization from task connectivity

The oldest strategy correlates connectivity within an a-priori system with performance. Its most durable results concern domain-general control systems: global connectivity of a lateral prefrontal region predicts both working memory performance and, selectively, fluid intelligence (Cole et al., 2012); frontoparietal regions shift their connectivity according to task demands, supporting a flexible-hub account of adaptive control (Cole et al., 2013); and the frontoparietal/multiple-demand and cingulo-opercular systems recur as correlates of executive function across the lifespan (Hausman et al., 2021). In the present corpus this family is numerically dominant for working memory and executive control — 56 unique records across those two themes — and is where cognitive-domain vocabulary is richest, but it is also the family most exposed to small samples and in-sample inference.

### 3.2 Topological correlation

Graph metrics computed per subject are regressed on cognitive scores. Examples in the corpus span episodic memory (Matyi et al., 2022), cognitive aging and topological efficiency of the structural connectome (Li et al., 2020), and a large body of clinical work (section 6). This design is cheap and interpretable, but suffers acutely from the reliability and sample-size problems in section 4; a global metric inherits noise from every edge it averages.

### 3.3 Out-of-sample predictive modeling

The current mainstream. Connectome-based predictive modeling (CPM) formalized a feature-selection → summarization → model → permutation-test pipeline usable without machine-learning expertise (Shen et al., 2017), and the connectome-fingerprinting result that motivated it showed that individual FC profiles identify people across sessions and between task and rest, with frontoparietal edges most distinctive and the same networks most predictive of fluid intelligence (Finn et al., 2015). Landmark applications include a sustained-attention network predicting performance in unseen individuals and ADHD symptoms from rest in an independent paediatric sample (Rosenberg et al., 2015), generalization of that marker to reading recall (Jangraw et al., 2018), extension to older adults (Fountain-Zaragoza et al., 2018) and to neurodiverse youth (Horien et al., 2023), and a longitudinal demonstration in over 1,000 participants that age-14 attention-network strength predicts later cannabis and cigarette use (Weng et al., 2024). For general ability, a distributed resting-state network predicted about 20% of variance in a multi-task estimate of general intelligence in 884 HCP participants, with no single network necessary and redundant information spread across the brain (Dubois et al., 2018) — a finding consonant with the network-neuroscience theory of intelligence (Barbey, 2017). Executive function has been decomposed in the same idiom into common and specific components (Reineberg et al., 2014; Qu et al., 2025), and the approach has been extended to creativity (Beaty et al., 2018; Ovando-Tellez et al., 2022), mathematics (Lynn et al., 2022), language learning (Feng et al., 2021; Song et al., 2026), working memory in memory-impaired individuals (Avery et al., 2019; Yang et al., 2023) and cognitive control combining functional and structural edges (Lv et al., 2024).

Model class matters less than expected: deep networks, including connectome-specific architectures, perform comparably to kernel ridge regression across a wide range of behavioral and demographic targets in nearly 10,000 HCP and UK Biobank participants (He et al., 2019). Modality does matter: with regression method held constant, FC-based models beat anatomical T1 and diffusion models for behavior prediction in both HCP and ABCD, most clearly for the cognitive component (Ooi et al., 2022).

### 3.4 Multivariate latent modes

Rather than predicting one score, these methods find joint low-rank structure between connectivity and a battery. The canonical result is a single "positive–negative" population mode linking connectivity to lifestyle, demographic and psychometric measures across 461 HCP subjects and 280 variables (Smith et al., 2015); a pre-registered replication has since examined its robustness (Goyal et al., 2022). The same logic recurs in partial-least-squares analyses of post-stroke deficits across motor, attention, language and memory domains (Wu et al., 2025) and in population-scale association atlases — 647 traits against 64,620 FC measures in over 40,000 UK Biobank participants, with results exposed through a public browser (Zhao et al., 2023, *Imaging Neuroscience*). The interpretive hazard is the mirror image of section 3.1: a dominant latent mode is easy to over-read as a cognitive construct when it may index socioeconomic position, head motion, or general ability.

### 3.5 Lesion, disconnection, and stimulation

Causal and quasi-causal evidence. Lesion network mapping localizes symptoms to networks rather than to lesion sites (Boes et al., 2015). Disconnectome approaches estimate which tracts a lesion severs and relate that to deficits: phonemic versus semantic fluency in 1,231 stroke patients (Biesbroek et al., 2021), reading networks (Forkel et al., 2022), and post-stroke cognitive impairment (Zhou et al., 2026). A methodologically important recent result shows that domain-general ability inflates apparent overlap between domain-specific lesion-deficit maps: partitioning out *g* in 473 patients made maps for visuospatial ability, processing speed, language and anterograde memory significantly less spatially inter-correlated (Sozer et al., 2026). Stimulation closes the loop — connectivity-defined targets for rTMS in depression (Elbau et al., 2023) and personalized stimulation of memory networks (Cash et al., 2022) treat the network map as an intervention plan rather than a description.

---

## 4. What determines whether a mapping replicates

This is the most consequential methodological thread in the corpus, and it should shape any new project's design.

1. **Effect sizes are small and samples have been far too small.** Across roughly 50,000 individuals in three large datasets, brain-wide association effects were smaller than previously assumed, with inflated effect sizes and replication failures at typical sample sizes; replication rates improved as samples grew into the thousands. Notably, effects were more robust for functional than structural MRI, for cognitive tests than for mental-health questionnaires, and for multivariate than univariate methods (Marek et al., 2022). The counterpoint argues that careful design — reliable measurement, state manipulation, multivariate targets — can establish robust brain–behavior relationships without thousands of participants (Rosenberg & Finn, 2022).
2. **Edge-level reliability is poor.** A meta-analysis of a decade of test–retest studies put the mean edge-level ICC at 0.29 (95% CI 0.23–0.36), with reliability higher for stronger within-network cortical edges, awake eyes-open acquisition, more within-subject data, and shorter retest intervals (Noble et al., 2019); see also Tozzi et al. (2020) for the within-connectome distribution of reliability. Multivariate summaries are more reliable than individual edges (Yoo et al., 2019), which is part of why prediction works better than edge-wise inference. Fingerprinting-based reliability metrics are now used as a QC layer in their own right (Ramduny et al., 2024).
3. **Motion and physiology create systematic, not random, error.** Subject motion induces distance-dependent, systematically structured correlations in FC matrices (Power et al., 2012) — an artifact that mimics exactly the short-range/long-range contrasts often reported as cognitive effects.
4. **Analytic degrees of freedom are large.** Transparency and pre-registration norms were formalized for neuroimaging generally (Poldrack et al., 2017), with ten concrete rules for individual-difference prediction specifically — nested cross-validation, no feature selection outside the training fold, permutation-based inference, and reporting of null-model baselines (Scheinost et al., 2019).
5. **Reference distributions are needed.** Lifespan normative modelling now provides them: brain charts for structural phenotypes across the lifespan (Bethlehem et al., 2022) and functional-connectome growth curves from 33,250 individuals aged 32 postmenstrual weeks to 80 years across 132 sites, with system-specific maturation timelines organized along a sensorimotor-to-association axis (Sun et al., 2025). Normative deviation scores are already being used to detect abnormal connectivity in psychiatric cohorts (Oliveira-Saraiva et al., 2023).

---

## 5. The interpretation problem

Even a replicable, well-powered mapping faces three validity questions the corpus treats unevenly.

**What is the cognitive variable?** Scores from any complex task load on *g*. Unless general ability is modelled explicitly, a "working memory network" may be a general-ability network — the lesion-mapping demonstration of this is the clearest in the corpus (Sozer et al., 2026), and the distributed, redundant character of *g*-predictive networks (Dubois et al., 2018; Buchanan et al., 2026, preprint, 38,824 individuals across three cohorts) makes the confound structural rather than incidental. Ontology-guided feature construction is one answer: integrating a cognitive ontology with connectivity analysis improved mean prediction accuracy to 0.13 across 16 cognitive tasks versus 0.08 for conventional approaches, while performing no better than conventional methods for sensory, motor and emotional tasks (0.03 versus 0.04) — a dissociation suggesting a domain-general association-cortex "cognitive connectome" (Wu et al., 2024).

**What is the brain variable?** Reverse inference from a network to a function requires a mapping from functions to networks estimated independently. Automated meta-analysis supplies it — large-scale synthesis of the activation literature (Yarkoni et al., 2011) and its successor based on comprehensive text-to-map prediction (Dockès et al., 2020) — as do co-activation-based characterizations of regional function (Genon et al., 2018). Network nomenclature itself is unstandardized, a problem flagged explicitly in the corpus (Uddin et al., 2019); two papers using "frontoparietal network" may not refer to the same vertices.

**What does a predictive feature mean?** A model that predicts well may rely on confounds or on redundant global signal. Early decoding work already showed whole-brain connectivity patterns distinguish cognitive states (Shirer et al., 2012) and predict maturational age (Dosenbach et al., 2010), both demonstrations that prediction accuracy certifies information content, not mechanism. Genetic anchoring offers partial external validation: GWAS of 206 structural connectivity measures in 26,333 UK Biobank participants implicated myelination and axon-guidance genes (Wainberg et al., 2024).

---

## 6. Clinical and translational strand

Roughly a third of the corpus is clinical, and it mostly asks whether network disruption explains cognitive impairment.

- **Alzheimer's disease and MCI** — multimodal Bayesian meta-analysis of graph-theoretic characteristics across the AD spectrum (Liu et al., 2025) and gradient-based evidence for altered functional segregation in amnestic MCI and AD (He et al., 2023).
- **Multiple sclerosis** — network controllability of the thalamus as a predictor of cognitive impairment (Yang et al., 2025), with the broader cognitive picture reviewed by De Meo and colleagues (2026).
- **Epilepsy** — connectome biomarkers of drug resistance (Larivière et al., 2020).
- **Tumours** — structural connectome-based prediction of cognitive deficits in treated glioma (Friedrich et al., 2023).
- **Stroke** — the disconnection and lesion-network literature of section 3.5.
- **Psychiatry** — connectome gradient dysfunction in major depression, linked to gene expression and treatment outcome (Xia et al., 2022).

Three cohort resources underpin most of the non-clinical work and are worth naming explicitly as the field's shared substrate: the HCP (Van Essen et al., 2013), UK Biobank imaging (Miller et al., 2016), and ABCD (Casey et al., 2018).

---

## 7. Machine-learning frontier

Graph-structured models are the active methodological edge: convolutional architectures designed for connectome topology (Kawahara et al., 2016), interpretable graph neural networks with region-selection layers (Li et al., 2021), GNN approaches to intelligence from resting-state fMRI (Thapaliya et al., 2024), and models of cognitive processes explicitly constrained by human connectomes (Zhang et al., 2022). The honest summary from the corpus is that architectural sophistication has not yet bought accuracy: kernel regression remains competitive at all sample sizes tested (He et al., 2019). The plausible value of these models lies elsewhere — in interpretability, in multimodal fusion, and in transfer across cohorts — and that is not yet demonstrated at scale.

---

## 8. Gaps, and what this project could do

1. **Construct-level mapping is underdeveloped relative to score-level prediction.** Most papers predict a task score; few model the latent cognitive structure (a bifactor or correlated-factors model of *g* plus domain factors) and ask which connectome features map to which factor. The two corpus papers that do this well (Reineberg et al., 2014; Sozer et al., 2026) are a decade apart and use different modalities.
2. **The domain-general/domain-specific dissociation is unresolved.** Independent lines — redundant distributed *g* networks (Dubois et al., 2018), association-cortex cognitive connectomes (Wu et al., 2024), and *g*-partitioned lesion maps (Sozer et al., 2026) — point the same way without having been reconciled in one analysis.
3. **Representation comparisons are rarely like-for-like.** Modality (Ooi et al., 2022), state (Zhao et al., 2023; Finn & Bandettini, 2021), parcellation (Kong et al., 2021) and model class (He et al., 2019) have each been benchmarked separately, almost never jointly within one cohort and target set.
4. **Reliability is reported inconsistently.** With mean edge ICC near 0.29 (Noble et al., 2019), any new edge-level claim needs an attenuation-corrected or multivariate treatment as standard, not as a robustness appendix.
5. **Nomenclature and ontology.** Cross-study synthesis is blocked by inconsistent network naming (Uddin et al., 2019) and the absence of a shared cognitive ontology layer in most analyses.

A concrete first empirical step for *neuropsychologygraphs*, consistent with the methodological consensus above: in one open cohort with both diffusion and multi-state fMRI (HCP or ABCD), fit a latent cognitive model, then predict the general factor and domain-specific residuals from a factorial crossing of edge type (FC, SC, structure–function coupling), state (rest, task, movie), and parcellation granularity, under a single nested cross-validation protocol with permutation inference and explicit motion and site handling. That design answers questions 1–4 simultaneously and produces a reusable benchmark rather than another point estimate.

---

## 9. Method of this review

PubMed was queried through the NCBI E-utilities connector on 2026-10-22 with 23 thematic queries (restricted to 2005 onward, relevance-sorted, up to 30 records each) spanning graph-theoretic foundations, CPM, intelligence, working memory, attention, episodic memory, language, executive control, structural connectomes, dynamic connectivity, gradients, CCA/PLS, reproducibility, reliability, precision fMRI, deep learning, clinical cognition, lesion networks, meta-analytic connectivity, parcellation, task-versus-rest, and development/aging; 589 unique records were hydrated with full metadata. Because relevance-sorted queries under-retrieve older field-defining work, 53 anchor papers were additionally sought by title lookup, with up to three hits per title retained and each candidate title-matched against the intended reference: 10 titles returned no PubMed match and 2 returned only loose matches, both discarded, and the lookups contributed 83 further records. The yearly counts in the figure are PubMed `total_count` values for one fixed query evaluated year by year. OpenAlex citation-graph enrichment was attempted but requires an API key not configured in this workspace, so no citation counts are reported here; preprint and journal versions of the same work were deduplicated in favour of the journal version. This is a narrative review of a convenience corpus, not a systematic review: no PRISMA screening, no inclusion/exclusion adjudication by a second reader, and no quality appraisal were performed.

**Corpus composition** (records; themes overlap): working memory and executive control 60, attention 30, episodic memory 30, language/semantics 30, structural connectome and cognition 30, dynamic connectivity 30, gradients 30, precision fMRI 30, deep learning 30, clinical cognition 30, lesion/disconnection 30, parcellation 30, task-versus-rest 30, reliability 30, development/aging 46, meta-analytic connectivity 18, CCA/PLS 11, reproducibility 5.

---

## References

1. Mesulam MM (1998). From sensation to cognition. *Brain*. PMID 9648540. doi:10.1093/brain/121.6.1013
2. Bullmore E, Sporns O (2009). Complex brain networks: graph theoretical analysis of structural and functional systems. *Nat Rev Neurosci*. PMID 19190637. doi:10.1038/nrn2575
3. Dosenbach NU, et al. (2010). Prediction of individual brain maturity using fMRI. *Science*. PMID 20829489. doi:10.1126/science.1194144
4. Duncan J (2010). The multiple-demand (MD) system of the primate brain: mental programs for intelligent behaviour. *Trends Cogn Sci*. PMID 20171926. doi:10.1016/j.tics.2010.01.004
5. Power JD, et al. (2011). Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. *Neuroimage*. PMID 22019881. doi:10.1016/j.neuroimage.2011.10.018
6. Shirer WR, et al. (2011). Decoding subject-driven cognitive states with whole-brain connectivity patterns. *Cereb Cortex*. PMID 21616982. doi:10.1093/cercor/bhr099
7. Yarkoni T, et al. (2011). Large-scale automated synthesis of human functional neuroimaging data. *Nat Methods*. PMID 21706013. doi:10.1038/nmeth.1635
8. Yeo BT, et al. (2011). The organization of the human cerebral cortex estimated by intrinsic functional connectivity. *J Neurophysiol*. PMID 21653723. doi:10.1152/jn.00338.2011
9. Cole MW, et al. (2012). Global connectivity of prefrontal cortex predicts cognitive control and intelligence. *J Neurosci*. PMID 22745498. doi:10.1523/JNEUROSCI.0536-12.2012
10. Cole MW, et al. (2013). Multi-task connectivity reveals flexible hubs for adaptive task control. *Nat Neurosci*. PMID 23892552. doi:10.1038/nn.3470
11. Van Essen DC, et al. (2013). The WU-Minn Human Connectome Project: an overview. *Neuroimage*. PMID 23684880. doi:10.1016/j.neuroimage.2013.05.041
12. Reineberg AE, et al. (2014). Resting-state networks predict individual differences in common and specific aspects of executive function. *Neuroimage*. PMID 25281800. doi:10.1016/j.neuroimage.2014.09.045
13. Boes AD, et al. (2015). Network localization of neurological symptoms from focal brain lesions. *Brain*. PMID 26264514. doi:10.1093/brain/awv228
14. Finn ES, et al. (2015). Functional connectome fingerprinting: identifying individuals using patterns of brain connectivity. *Nat Neurosci*. PMID 26457551. doi:10.1038/nn.4135
15. Medaglia JD, et al. (2015). Cognitive network neuroscience. *J Cogn Neurosci*. PMID 25803596. doi:10.1162/jocn_a_00810
16. Rosenberg MD, et al. (2015). A neuromarker of sustained attention from whole-brain functional connectivity. *Nat Neurosci*. PMID 26595653. doi:10.1038/nn.4179
17. Smith SM, et al. (2015). A positive-negative mode of population covariation links brain connectivity, demographics and behavior. *Nat Neurosci*. PMID 26414616. doi:10.1038/nn.4125
18. Betzel RF, Bassett DS (2016). Multi-scale brain networks. *Neuroimage*. PMID 27845257. doi:10.1016/j.neuroimage.2016.11.006
19. Glasser MF, et al. (2016). A multi-modal parcellation of human cerebral cortex. *Nature*. PMID 27437579. doi:10.1038/nature18933
20. Kawahara J, et al. (2016). BrainNetCNN: Convolutional neural networks for brain networks; towards predicting neurodevelopment. *Neuroimage*. PMID 27693612. doi:10.1016/j.neuroimage.2016.09.046
21. Margulies DS, et al. (2016). Situating the default-mode network along a principal gradient of macroscale cortical organization. *Proc Natl Acad Sci U S A*. PMID 27791099. doi:10.1073/pnas.1608282113
22. Miller KL, et al. (2016). Multimodal population brain imaging in the UK Biobank prospective epidemiological study. *Nat Neurosci*. PMID 27643430. doi:10.1038/nn.4393
23. Avena-Koenigsberger A, et al. (2017). Communication dynamics in complex brain networks. *Nat Rev Neurosci*. PMID 29238085. doi:10.1038/nrn.2017.149
24. Barbey AK (2017). Network Neuroscience Theory of Human Intelligence. *Trends Cogn Sci*. PMID 29167088. doi:10.1016/j.tics.2017.10.001
25. Gordon EM, et al. (2017). Precision Functional Mapping of Individual Human Brains. *Neuron*. PMID 28757305. doi:10.1016/j.neuron.2017.07.011
26. Jangraw DC, et al. (2017). A functional connectivity-based neuromarker of sustained attention generalizes to predict recall in a reading task. *Neuroimage*. PMID 29031531. doi:10.1016/j.neuroimage.2017.10.019
27. Poldrack RA, et al. (2017). Scanning the horizon: towards transparent and reproducible neuroimaging research. *Nat Rev Neurosci*. PMID 28053326. doi:10.1038/nrn.2016.167
28. Shen X, et al. (2017). Using connectome-based predictive modeling to predict individual behavior from brain connectivity. *Nat Protoc*. PMID 28182017. doi:10.1038/nprot.2016.178
29. Beaty RE, et al. (2018). Robust prediction of individual creative ability from brain functional connectivity. *Proc Natl Acad Sci U S A*. PMID 29339474. doi:10.1073/pnas.1713532115
30. Casey BJ, et al. (2018). The Adolescent Brain Cognitive Development (ABCD) study: Imaging acquisition across 21 sites. *Dev Cogn Neurosci*. PMID 29567376. doi:10.1016/j.dcn.2018.03.001
31. Dubois J, et al. (2018). A distributed brain network predicts general intelligence from resting-state human neuroimaging data. *Philos Trans R Soc Lond B Biol Sci*. PMID 30104429. doi:10.1098/rstb.2017.0284
32. Eickhoff SB, et al. (2018). Imaging-based parcellations of the human brain. *Nat Rev Neurosci*. PMID 30305712. doi:10.1038/s41583-018-0071-7
33. Fountain-Zaragoza S, et al. (2018). Connectome-based models predict attentional control in aging adults. *Neuroimage*. PMID 30394324. doi:10.1016/j.neuroimage.2018.10.074
34. Genon S, et al. (2018). How to Characterize the Function of a Brain Region. *Trends Cogn Sci*. PMID 29501326. doi:10.1016/j.tics.2018.01.010
35. Gratton C, et al. (2018). Functional Brain Networks Are Dominated by Stable Group and Individual Factors, Not Cognitive or Daily Variation. *Neuron*. PMID 29673485. doi:10.1016/j.neuron.2018.03.035
36. Greene AS, et al. (2018). Task-induced brain state manipulation improves prediction of individual traits. *Nat Commun*. PMID 30022026. doi:10.1038/s41467-018-04920-3
37. Schaefer A, et al. (2018). Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic Functional Connectivity MRI. *Cereb Cortex*. PMID 28981612. doi:10.1093/cercor/bhx179
38. Avery EW, et al. (2019). Distributed Patterns of Functional Connectivity Predict Working Memory Performance in Novel Healthy and Memory-impaired Individuals. *J Cogn Neurosci*. PMID 31659926. doi:10.1162/jocn_a_01487
39. Baum GL, et al. (2019). Development of structure-function coupling in human brain networks during youth. *Proc Natl Acad Sci U S A*. PMID 31874926. doi:10.1073/pnas.1912034117
40. Elliott ML, et al. (2019). General functional connectivity: Shared features of resting-state and task fMRI drive reliable and heritable individual differences in functional brain networks. *Neuroimage*. PMID 30708106. doi:10.1016/j.neuroimage.2019.01.068
41. Farahani FV, et al. (2019). Application of Graph Theory for Identifying Connectivity Patterns in Human Brain Networks: A Systematic Review. *Front Neurosci*. PMID 31249501. doi:10.3389/fnins.2019.00585
42. He T, et al. (2019). Deep neural networks and kernel regression achieve comparable accuracies for functional connectivity prediction of behavior and demographics. *Neuroimage*. PMID 31610298. doi:10.1016/j.neuroimage.2019.116276
43. Kong R, et al. (2019). Spatial Topography of Individual-Specific Cortical Networks Predicts Human Cognition, Personality, and Emotion. *Cereb Cortex*. PMID 29878084. doi:10.1093/cercor/bhy123
44. Noble S, et al. (2019). A decade of test-retest reliability of functional connectivity: A systematic review and meta-analysis. *Neuroimage*. PMID 31494250. doi:10.1016/j.neuroimage.2019.116157
45. Scheinost D, et al. (2019). Ten simple rules for predictive modeling of individual differences in neuroimaging. *Neuroimage*. PMID 30831310. doi:10.1016/j.neuroimage.2019.02.057
46. Uddin LQ, et al. (2019). Towards a Universal Taxonomy of Macro-scale Functional Human Brain Networks. *Brain Topogr*. PMID 31707621. doi:10.1007/s10548-019-00744-6
47. Yoo K, et al. (2019). Multivariate approaches improve the reliability and validity of functional connectivity and prediction of individual behaviors. *Neuroimage*. PMID 31039408. doi:10.1016/j.neuroimage.2019.04.060
48. Cui Z, et al. (2020). Individual Variation in Functional Topography of Association Networks in Youth. *Neuron*. PMID 32078800. doi:10.1016/j.neuron.2020.01.029
49. Dockès J, et al. (2020). NeuroQuery, comprehensive meta-analysis of human brain mapping. *Elife*. PMID 32129761. doi:10.7554/eLife.53385
50. Larivière S, et al. (2020). Connectome biomarkers of drug-resistant epilepsy. *Epilepsia*. PMID 33236784. doi:10.1111/epi.16753
51. Li X, et al. (2020). Age-Related Decline in the Topological Efficiency of the Brain Structural Connectome and Cognitive Aging. *Cereb Cortex*. PMID 32219315. doi:10.1093/cercor/bhaa066
52. Suárez LE, et al. (2020). Linking Structure and Function in Macroscale Brain Networks. *Trends Cogn Sci*. PMID 32160567. doi:10.1016/j.tics.2020.01.008
53. Tian Y, et al. (2020). Topographic organization of the human subcortex unveiled with functional connectivity gradients. *Nat Neurosci*. PMID 32989295. doi:10.1038/s41593-020-00711-6
54. Tozzi L, et al. (2020). Test-retest reliability of the human functional connectome over consecutive days: identifying highly reliable portions and assessing the impact of methodological choices. *Netw Neurosci*. PMID 33615097. doi:10.1162/netn_a_00148
55. Biesbroek JM, et al. (2021). Anatomy of phonemic and semantic fluency: A lesion and disconnectome study in 1231 stroke patients. *Cortex*. PMID 34450565. doi:10.1016/j.cortex.2021.06.019
56. Feng G, et al. (2021). Neural Fingerprints Underlying Individual Language Learning Profiles. *J Neurosci*. PMID 34301824. doi:10.1523/JNEUROSCI.0415-21.2021
57. Finn ES, Bandettini PA (2021). Movie-watching outperforms rest for functional connectivity-based prediction of behavior. *Neuroimage*. PMID 33813007. doi:10.1016/j.neuroimage.2021.117963
58. Hausman HK, et al. (2021). Cingulo-opercular and frontoparietal control network connectivity and executive functioning in older adults. *Geroscience*. PMID 34950997. doi:10.1007/s11357-021-00503-1
59. Kong R, et al. (2021). Individual-Specific Areal-Level Parcellations Improve Functional Connectivity Prediction of Behavior. *Cereb Cortex*. PMID 33942058. doi:10.1093/cercor/bhab101
60. Li X, et al. (2021). BrainGNN: Interpretable Brain Graph Neural Network for fMRI Analysis. *Med Image Anal*. PMID 34655865. doi:10.1016/j.media.2021.102233
61. Smallwood J, et al. (2021). The default mode network in cognition: a topographical perspective. *Nat Rev Neurosci*. PMID 34226715. doi:10.1038/s41583-021-00474-4
62. Bethlehem RAI, et al. (2022). Brain charts for the human lifespan. *Nature*. PMID 35388223. doi:10.1038/s41586-022-04554-y
63. Cash RFH, et al. (2022). Personalized brain stimulation of memory networks. *Brain Stimul*. PMID 36113762. doi:10.1016/j.brs.2022.09.004
64. Forkel SJ, et al. (2022). Stroke disconnectome decodes reading networks. *Brain Struct Funct*. PMID 36192557. doi:10.1007/s00429-022-02575-x
65. Goyal N, et al. (2022). The positive-negative mode link between brain connectivity, demographics and behaviour: a pre-registered replication of Smith. (2015). *R Soc Open Sci*. PMID 35186306. doi:10.1098/rsos.201090
66. Lynn A, et al. (2022). Predicting children's math skills from task-based and resting-state functional brain connectivity. *Cereb Cortex*. PMID 34974615. doi:10.1093/cercor/bhab476
67. Marek S, et al. (2022). Reproducible brain-wide association studies require thousands of individuals. *Nature*. PMID 35296861. doi:10.1038/s41586-022-04492-9
68. Matyi MA, Spielberg JM (2022). The structural brain network topology of episodic memory. *PLoS One*. PMID 35749536. doi:10.1371/journal.pone.0270592
69. Ooi LQR, et al. (2022). Comparison of individualized behavioral predictions across anatomical, diffusion and functional connectivity MRI. *Neuroimage*. PMID 36116616. doi:10.1016/j.neuroimage.2022.119636
70. Ovando-Tellez M, et al. (2022). Brain connectivity-based prediction of real-life creativity is mediated by semantic memory structure. *Sci Adv*. PMID 35119928. doi:10.1126/sciadv.abl4294
71. Rosenberg MD, Finn ES (2022). How to establish robust brain-behavior relationships without thousands of individuals. *Nat Neurosci*. PMID 35710985. doi:10.1038/s41593-022-01110-9
72. Xia M, et al. (2022). Connectome gradient dysfunction in major depression and its association with gene expression profiles and treatment outcomes. *Mol Psychiatry*. PMID 35338312. doi:10.1038/s41380-022-01519-5
73. Zhang Y, et al. (2022). Deep learning models of cognitive processes constrained by human brain connectomes. *Med Image Anal*. PMID 35738052. doi:10.1016/j.media.2022.102507
74. Elbau IG, et al. (2023). Functional Connectivity Mapping for rTMS Target Selection in Depression. *Am J Psychiatry*. PMID 36855880. doi:10.1176/appi.ajp.20220306
75. Friedrich M, et al. (2023). Structural connectome-based predictive modeling of cognitive deficits in treated glioma patients. *Neurooncol Adv*. PMID 38196739. doi:10.1093/noajnl/vdad151
76. He Y, et al. (2023). Functional gradients reveal altered functional segregation in patients with amnestic mild cognitive impairment and Alzheimer's disease. *Cereb Cortex*. PMID 37718155. doi:10.1093/cercor/bhad328
77. Horien C, et al. (2023). A generalizable connectome-based marker of in-scan sustained attention in neurodiverse youth. *Cereb Cortex*. PMID 36573438. doi:10.1093/cercor/bhac506
78. Oliveira-Saraiva D, Ferreira HA (2023). Normative model detects abnormal functional connectivity in psychiatric disorders. *Front Psychiatry*. PMID 36873218. doi:10.3389/fpsyt.2023.1068397
79. Yang H, et al. (2023). Using modular connectome-based predictive modeling to reveal brain-behavior relationships of individual differences in working memory. *Brain Struct Funct*. PMID 37349540. doi:10.1007/s00429-023-02666-3
80. Zhao B, et al. (2023). An atlas of trait associations with resting-state and task-evoked human brain functional organizations in the UK Biobank. *Imaging Neurosci (Camb)*. PMID 38770197. doi:10.1162/imag_a_00015
81. Zhao W, et al. (2023). Task fMRI paradigms may capture more behaviorally relevant information than resting-state functional connectivity. *Neuroimage*. PMID 36801369. doi:10.1016/j.neuroimage.2023.119946
82. Hardikar S, et al. (2024). Macro-scale patterns in functional connectivity associated with ongoing thought patterns and dispositional traits. *Elife*. PMID 39565648. doi:10.7554/eLife.93689
83. Hermosillo RJM, et al. (2024). A precision functional atlas of personalized network topography and probabilities. *Nat Neurosci*. PMID 38532024. doi:10.1038/s41593-024-01596-5
84. Kraljević N, et al. (2024). Network and state specificity in connectivity-based predictions of individual behavior. *Hum Brain Mapp*. PMID 38864353. doi:10.1002/hbm.26753
85. Lv Q, et al. (2024). Connectome-based prediction modeling of cognitive control using functional and structural connectivity. *Brain Cogn*. PMID 39250856. doi:10.1016/j.bandc.2024.106221
86. Ramduny J, Kelly C (2024). Connectome-based fingerprinting: reproducibility, precision, and behavioral prediction. *Neuropsychopharmacology*. PMID 39147868. doi:10.1038/s41386-024-01962-8
87. Thapaliya B, et al. (2024). Brain networks and intelligence: A graph neural network based approach to resting state fMRI data. *Med Image Anal*. PMID 39708510. doi:10.1016/j.media.2024.103433
88. Wainberg M, et al. (2024). Genetic architecture of the structural connectome. *Nat Commun*. PMID 38438384. doi:10.1038/s41467-024-46023-2
89. Weng Y, et al. (2024). A robust brain network for sustained attention from adolescence to adulthood that predicts later substance use. *Elife*. PMID 39235858. doi:10.7554/eLife.97150
90. Wu G, et al. (2024). Unveiling the core functional networks of cognition: An ontology-guided machine learning approach. *Neuroimage*. PMID 39173695. doi:10.1016/j.neuroimage.2024.120804
91. Gopalakrishnan V, et al. (2025). Multiscale comparative connectomics. *Imaging Neurosci (Camb)*. PMID 40800798. doi:10.1162/IMAG.a.2
92. Liu W, et al. (2025). The whole-brain structural and functional connectome in Alzheimer's disease spectrum: A multimodal Bayesian meta-analysis of graph theoretical characteristics. *Neurosci Biobehav Rev*. PMID 40280288. doi:10.1016/j.neubiorev.2025.106174
93. Nordin K, et al. (2025). Two long-axis dimensions of hippocampal-cortical integration support memory function across the adult lifespan. *Elife*. PMID 40110999. doi:10.7554/eLife.97658
94. Qu S, et al. (2025). Connectome-Based Predictive Models of General and Specific Executive Functions. *Hum Brain Mapp*. PMID 40956153. doi:10.1002/hbm.70358
95. Sun L, et al. (2025). Human lifespan changes in the brain's functional connectome. *Nat Neurosci*. PMID 40181189. doi:10.1038/s41593-025-01907-4
96. Wu K, et al. (2025). Latent dimension linking functional connectivity with post-stroke deficits across multiple domains. *Brain Commun*. PMID 40740434. doi:10.1093/braincomms/fcaf276
97. Yang Y, et al. (2025). Thalamic Network Controllability Predicts Cognitive Impairment in Multiple Sclerosis. *Hum Brain Mapp*. PMID 40637347. doi:10.1002/hbm.70284
98. Buchanan CR, et al. (2026). General cognitive function and the brain's structural connectome. *bioRxiv*. PMID 41959038. doi:10.64898/2026.03.30.715223
99. De Meo E, et al. (2026). Cognition in multiple sclerosis. *Curr Opin Neurol*. PMID 41960777. doi:10.1097/WCO.0000000000001481
100. Ghasemi N, et al. (2026). Structural Connectivity of Functionally Defined Episodic Memory Networks: A Large-Scale Connectome-Behavior Study. *Brain Behav*. PMID 42216780. doi:10.1002/brb3.71515
101. Song P, et al. (2026). Multi-network Topology Underlying Individual Language Learning Success. *J Neurosci*. PMID 42297561. doi:10.1523/JNEUROSCI.2205-25.2026
102. Sozer L, et al. (2026). Cognition disconnected: the influence of domain-general cognition (g) in lesion-deficit mapping. *Cereb Cortex*. PMID 42619488. doi:10.1093/cercor/bhag120
103. Zhou Y, et al. (2026). Lesion network mapping for post-stroke cognitive impairment. *Int J Stroke*. PMID 41644505. doi:10.1177/17474930261425249
