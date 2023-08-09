if (( "$#" == 0 )); then 
    ID=$(mkid)
else
    ID=$1
fi

echo "Running with id $ID"

CUBLAS_WORKSPACE_CONFIG=:4096:8 ebatch sd0  slconf "python -m train l_bad_base=0 l_bad_adapted=0.01 max_adapt_steps=5 adversary.early_stop=False seed=0"
CUBLAS_WORKSPACE_CONFIG=:4096:8 ebatch sd0_ slconf "python -m train l_bad_base=0 l_bad_adapted=0.01 max_adapt_steps=5 adversary.early_stop=False seed=0"
CUBLAS_WORKSPACE_CONFIG=:4096:8 ebatch sd1  slconf "python -m train l_bad_base=0 l_bad_adapted=0.01 max_adapt_steps=5 adversary.early_stop=False seed=1"
CUBLAS_WORKSPACE_CONFIG=:4096:8 ebatch sd2  slconf "python -m train l_bad_base=0 l_bad_adapted=0.01 max_adapt_steps=5 adversary.early_stop=False seed=2"
CUBLAS_WORKSPACE_CONFIG=:4096:8 ebatch sd3  slconf "python -m train l_bad_base=0 l_bad_adapted=0.01 max_adapt_steps=5 adversary.early_stop=False seed=3"


