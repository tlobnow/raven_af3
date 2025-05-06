# User parameters for AlphaFold3 on MPCDF systems, common to MSA and inference jobs.

# We need a local directory that contains the model weights and a cache directory for JAX
# (keep as-is if unsure)
AF3_USER_DIR=${HOME}/alphafold_3_0_1
AF3_MODEL_DIR=${AF3_USER_DIR}/model
AF3_JAX_CACHE_DIR=${AF3_USER_DIR}/jaxcache

