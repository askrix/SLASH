# Source folder quick overview:   

1. `EinsumNetwork`, `NeurASP` and `tabnet` contain code which was cloned from the [git repository EinNets](https://github.com/cambridge-mlg/EinsumNetworks), [git repository NeurASP](https://github.com/azreasoners/NeurASP) and [git repository Tabnet](https://github.com/dreamquark-ai/tabnet).
2. `SLASH` contains all modifications made to enable probabilisitc circuits in the program and to use the SlotAttention Module.
    * `mvpp.py` contains all functions to compute stable models given a logic program and to compute the gradients using the output probabilites  
    * `slash.py` contains the SLASH class which brings together the symbolic program with basic neural nets or probabilistic circuits
3. `SlotAttentionObjDiscovery` contains an example implemention of the ObjectDiscovery Task on the Shapeworld dataset.
4. `SlotAttentionSetPrediction` contains an example implemention of the Setprediction Task on the Shapeworld dataset.

4. `slash*` folders contain the different experiment setups
- `slash_mnist_digit_addition` digit addition experiment setup including missing data imputation
- `slash_slot_attention_shapeworld2` slot attention experiment setup for the Shapeworld dataset with exactly two objects consisting of 3 colors and 3 shapes
- `slash_slot_attention_shapeworld4` slot attention experiment setup for Shapeworld with 4 concepts and 2-4 objects

- `slash_slot_attention_cogent` slot attention experiment following the CoGenT test described in the CLEVR paper.
- `slash_slot_attention_ood` slot attention experiment for out of distribution testing


