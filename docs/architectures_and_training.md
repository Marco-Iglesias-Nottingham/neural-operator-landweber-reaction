# Architectures and training

Four headline architectures are used:

- FNO (`models/fno2d_standard.py`);
- standard DeepONet (`models/deeponet_standard.py`);
- GenFNO (`models/fno2d_generalized.py`); and
- GenDeepONet (`models/deeponet_generalized.py`).

The frozen models, derivative weights and checkpoint epochs are declared in
`configs/paper/downstream_models.tsv`. Selection uses validation `E_F`, `E_J`
and their combined score; independent test errors are evaluated only after
freezing. Training is capped at 2,500 epochs. Exact CLI arguments and campaign
membership are in `configs/paper/training_campaign.tsv`.
