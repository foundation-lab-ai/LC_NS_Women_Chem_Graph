# A mass-difference graph resolves a pre-diagnostic lung cancer lipid signature into a desaturation and chain-elongation series in never-smoking women

GitHub repository for the project.

## Authors

- Maria Vaida<sup>1†</sup>
- Ziyuan Huang<sup>2†*</sup>

<sup>1</sup> School of Analytics & Computational Sciences, Harrisburg University of Science and Technology, Harrisburg, Pennsylvania, United States of America  
<sup>2</sup> Department of Emergency Medicine and Department of Microbiology, University of Massachusetts Chan Medical School, Worcester, Massachusetts, United States of America

† These authors contributed equally to this work.  
* Corresponding author: ziyuan.huang2@umassmed.edu

## Abstract

Untargeted mass spectrometry measures tens of thousands of features in human plasma, and most cannot be matched to a named compound. We propose a graph that requires no compound names at all. Every feature is a node, and two features are joined when the difference between their measured masses equals the mass of a known biochemical reaction to within 5 parts per million. An edge is therefore consistent with one reaction step whatever the compounds are.

We applied the graph to the public data of a study of lung cancer in never-smoking women, 838 samples and 15,292 features (Metabolomics Workbench ST002773). The study reported a module of 121 features associated with future lung cancer. Of the 90 negative-mode features, 64 nonredundant features remained after isotopologues were collapsed and 33 were connected by mass-difference edges. In the original negative-mode feature set, 45.6% were connected compared with 7.1% in mass-matched random sets, a 6.4-fold enrichment (*p* = 0.0005). Six lyso-glycerophospholipids formed one connected series ordered by desaturation and chain length. For the LPE homologous series, retention time was not used to build the graph and followed chain length and unsaturation with *R*<sup>2</sup> = 0.90. Propagating along the edges identified one species absent from the published module, matching LPE 22:2 to within 1.1 parts per million.

A second published module of 440 features behaved differently. Its associations were reproduced, with published and reconstructed association statistics correlated at 0.80 and 0.90 across the two chromatographic columns, yet its features connected at only 1.2 times the rate of matched random sets.

Six untargeted analyses of these data produced no finding at a false discovery rate of 0.20. The graph organizes chemical relationships and does not by itself detect association.
