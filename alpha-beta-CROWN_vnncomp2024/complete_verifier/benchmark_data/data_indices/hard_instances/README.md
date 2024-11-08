This is a private folder.

```bash
ALPHA_BETA_CROWN_MIP_DEBUG=1 python robustness_verifier.py --config exp_configs/full_dataset/cifar_oval_wide.yaml --device cpu --data_idx_file data/data_indices/hard_instances/cifar_oval_wide_hard.txt --complete_verifier mip --mip_threads 16 --mip_multi_proc 1
ALPHA_BETA_CROWN_MIP_DEBUG=1 python robustness_verifier.py --config exp_configs/full_dataset/cifar_oval_base.yaml --device cpu --data_idx_file data/data_indices/hard_instances/cifar_oval_base_hard.txt --complete_verifier mip --mip_threads 16 --mip_multi_proc 1
ALPHA_BETA_CROWN_MIP_DEBUG=1 python robustness_verifier.py --config exp_configs/full_dataset/cifar_oval_deep.yaml --device cpu --data_idx_file data/data_indices/hard_instances/cifar_oval_base_deep.txt --complete_verifier mip --mip_threads 16 --mip_multi_proc 1
```

