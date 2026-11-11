# C118: native weak-state supervision to train the representation
Triggered byC117: severalepochs of dense MSE leave train andunseen errors almost unchanged, whereas C106fitsindividualmaps with manyrepetitions. Test whether directstate supervision provides useful gradients to the terrain representation.
This is not an establishedrootcause. C104alreadytested state learning on pointfeatures; thisnewtest uses native2D learnedterrainfields and whole-pixel supervision, not a repeat ofC104.
Freeze C117factorized monitor-selected checkpoint only afteritsrunfinishes. Both arms addidentical randomlyinitialized6logithead (2bandsx3residualstates); sameparameters,parent,sampleorder118073,batch8,optimizer.
Control loss: officialmaskedbandMSE/25 only. New loss: sameMSE +.25classbalancedcrossentropy. Classes based onTRAIN residuals Y-C82 (<-5,between,+>=5), inverseprevalence computed onFIT with sameinclusionweights. Auxhead istrainingonly; deploymentstillpredictsthecontinuouscorrection, no threshold or fusiontuning.
8epochs max,3patience,min4epochs; choosebestjointmeanoriginal128monitor inclinitialC117parentandC82. Freeze before484confirmation; samepromotiongate>=.30bothpositiveCI/nativecheck; no allTRAINexpansionwithoutgate.
Record initialbackbone gradient norms fromMSE andCE onexact sameFITbatch to verify the intended intervention operates. This verifiesmechanics, not futuregeneralization.
Resources: physical1/3 ifverifiedidle; otherwise3/4whenfree;fullUUID viaCC launcher. No officialVAL. No C95.
