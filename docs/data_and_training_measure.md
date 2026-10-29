# Data and training measure

The broad training measure is a mixture of non-periodic Matérn coefficient
fields. Smoothness and length scale vary across the declared support, as do
the physical mean and standard deviation, subject to the frozen coefficient
bounds. Production uses 10,000 training, 2,500 validation and 2,500 reserved
independent-test fields. The final test set is never used for checkpoint
selection.

Each field carries one forward solution and 16 JVP labels. The paper's
derivative loss uses covariance-scaled physical perturbations from the frozen
input basis; these are not described as unit eigenvectors. Data generators and
their metadata contracts are in `scripts/data/` and `configs/paper/`.
