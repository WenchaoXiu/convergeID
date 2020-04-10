# convergeID Overview
convergeID is a python script (python 3) for lineage convergent event prediction. It uses time-series single-cell RNA-seq data which contains two time points. It is now a developmental version.
>>>>>>> origin/master

# Installation
> 1. Download python scripts from github
> 2. Import the main function

```python
import sys
sys.path.append(path+'/convergeID/src/')
import convergeID
```

# Quick Start
**Input file**: Time-series single cell RNA-seq data.

**Output file**: An inferenced lineage convergent event list ranked by convergent strength.

1. Data loading
```python
# Load expresion data which is separated by tab. The columns are gene, the rows are cells.
# Data has a column "cell.label" which describes cell cluster information.
data_before,before_label,data_after,after_label = convergeID.load_data(before_path, after_path)
```

2. Predict lineage convergent candidate
```python
conv_pair_candidate,all_conv_pair,Nmodel = convergeID.get_conv_pair(data_before, before_label, data_after)
```

3. Get ranked lineage convergent event list
```python
# Ranking lineage convergent candidate by convergent strength.
conv_pair = convergeID.rank_by_conv_strength(conv_pair_candidate, all_conv_pair, after_label, Nmodel)
```

