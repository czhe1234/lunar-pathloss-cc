# C120: isolate missing polarity information from the continuous power readout
C118 spatial risk classifier locates unreliable regions but confuses opposite residual signs; C119 pair supervision worsened fullrisk. Before any more fieldtraining, ask a simpler diagnostic: given that a valid pixel has |Y-C82|>=5, can a dedicated classifier infer its sign from inputs alone?
The true tail gate is an ORACLE conditioning for this diagnostic, not an available inferenceinput or deployable score. All features computed solely from H/TX and frozen TRAIN-trained models. No actualresidual,masks,pixelids or selection flags infeatures.
Compare CatBoost binary classifier on169C101precise features versus those169plus96C118 native latentfeatures, same data/depth9/lr.05/max2000trees/patience150. Same1984FIT/128monitor. Two bands trained separately. Onlymonitordiagnostic, do notconsume484confirmation for unpromoteddiagnostic.
CheckpointC118locationbest frozen by previouspowerRMSE. Nativefeatures extracted atsame1536FIT/2048monitor points; onlylabel-tail unionrows materialized tosavecost (selectiondoesnotalterfeaturevalues).
Readoutgate: both bands signaccuracy>=.80 and >=.08 aboveoldfeatures plus logloss drop>=.05 before considering a fullinput-only conditionalpowerdecoder. If not, no expandedtraining or claimednewversion.
AllCC,rawTRAINonly,GPUs1/3/4UUIDlauncher;officialVALnotread;C95closed.
