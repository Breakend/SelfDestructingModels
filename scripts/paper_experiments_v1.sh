if (( "$#" == 0 )); then 
    ID=$(python -c "import random; chars='qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890'; print(''.join([random.choice(chars) for _ in range(7)]))")
else
    ID=$1
fi

echo "Running batch of jobs with id $ID"

#############
### PETER ###
#############
# 4 separate jobs for 16-step bios variants
ebatch bios_lin_mi_16_gp slconf_jag "python -m train experiment=bios l_bad_adapted=0.0 l_linear_mi=1.0 l_bad_adapted_grad=1.0 max_adapt_steps=16 batch_hash=$ID"
ebatch bios_lin_mi_16    slconf_jag "python -m train experiment=bios l_bad_adapted=0.0 l_linear_mi=1.0 l_bad_adapted_grad=0.0 max_adapt_steps=16 batch_hash=$ID"
ebatch bios_maxl_16_gp   slconf_jag "python -m train experiment=bios l_bad_adapted=1.0 l_linear_mi=0.0 l_bad_adapted_grad=1.0 max_adapt_steps=16 batch_hash=$ID"
ebatch bios_maxl_16      slconf_jag "python -m train experiment=bios l_bad_adapted=1.0 l_linear_mi=0.0 l_bad_adapted_grad=0.0 max_adapt_steps=16 batch_hash=$ID"


# 4 separate jobs for 4-step bios variants
# ebatch bios_maxl_4_gp   slconf_jag "python -m train experiment=bios l_bad_adapted=1.0 l_linear_mi=0.0 l_bad_adapted_grad=1.0 max_adapt_steps=4 batch_hash=$ID"
############
### ERIC ###
############
ebatch bios_maxl_4      slconf_jag "python -m train experiment=bios l_bad_adapted=1.0 l_linear_mi=0.0 l_bad_adapted_grad=0.0 max_adapt_steps=4 batch_hash=$ID"
ebatch bios_lin_mi_4_gp slconf_jag "python -m train experiment=bios l_bad_adapted=0.0 l_linear_mi=1.0 l_bad_adapted_grad=1.0 max_adapt_steps=4 batch_hash=$ID"
ebatch bios_lin_mi_4    slconf_jag "python -m train experiment=bios l_bad_adapted=0.0 l_linear_mi=1.0 l_bad_adapted_grad=0.0 max_adapt_steps=4 batch_hash=$ID"


# stack 2 0-step bios jobs per GPU
ebatch bios_0_gp slconf_jag  "{ python -m train experiment=bios l_bad_adapted=1.0 l_linear_mi=0.0 l_bad_adapted_grad=1.0 max_adapt_steps=0 batch_hash=$ID; } & { python -m train experiment=bios l_bad_adapted=0.0 l_linear_mi=1.0 l_bad_adapted_grad=1.0 max_adapt_steps=0 batch_hash=$ID; }; wait"
ebatch bios_0    slconf_jag "{ python -m train experiment=bios l_bad_adapted=1.0 l_linear_mi=0.0 l_bad_adapted_grad=0.0 max_adapt_steps=0 batch_hash=$ID; } & { python -m train experiment=bios l_bad_adapted=0.0 l_linear_mi=1.0 l_bad_adapted_grad=0.0 max_adapt_steps=0 batch_hash=$ID; }; wait"


# stack all different k steps on a single GPU for each run type
ebatch regr_maxl_gp   slconf "{ python -m train experiment=regression l_bad_adapted=1.0 l_linear_mi=0.0 l_bad_adapted_grad=1.0 max_adapt_steps=16 batch_hash=$ID; } & { python -m train experiment=regression l_bad_adapted=1.0 l_linear_mi=0.0 l_bad_adapted_grad=1.0 max_adapt_steps=4  batch_hash=$ID; } & { python -m train experiment=regression l_bad_adapted=1.0 l_linear_mi=0.0 l_bad_adapted_grad=1.0 max_adapt_steps=0  batch_hash=$ID; }; wait"
ebatch regr_maxl      slconf "{ python -m train experiment=regression l_bad_adapted=1.0 l_linear_mi=0.0 l_bad_adapted_grad=0.0 max_adapt_steps=16 batch_hash=$ID; } & { python -m train experiment=regression l_bad_adapted=1.0 l_linear_mi=0.0 l_bad_adapted_grad=0.0 max_adapt_steps=4  batch_hash=$ID; } & { python -m train experiment=regression l_bad_adapted=1.0 l_linear_mi=0.0 l_bad_adapted_grad=0.0 max_adapt_steps=0  batch_hash=$ID; }; wait"
ebatch regr_lin_mi_gp slconf "{ python -m train experiment=regression l_bad_adapted=0.0 l_linear_mi=1.0 l_bad_adapted_grad=1.0 max_adapt_steps=16 batch_hash=$ID; } & { python -m train experiment=regression l_bad_adapted=0.0 l_linear_mi=1.0 l_bad_adapted_grad=1.0 max_adapt_steps=4  batch_hash=$ID; } & { python -m train experiment=regression l_bad_adapted=0.0 l_linear_mi=1.0 l_bad_adapted_grad=1.0 max_adapt_steps=0  batch_hash=$ID; }; wait"
ebatch regr_lin_mi    slconf "{ python -m train experiment=regression l_bad_adapted=0.0 l_linear_mi=1.0 l_bad_adapted_grad=0.0 max_adapt_steps=16 batch_hash=$ID; } & { python -m train experiment=regression l_bad_adapted=0.0 l_linear_mi=1.0 l_bad_adapted_grad=0.0 max_adapt_steps=4  batch_hash=$ID; } & { python -m train experiment=regression l_bad_adapted=0.0 l_linear_mi=1.0 l_bad_adapted_grad=0.0 max_adapt_steps=0  batch_hash=$ID; }; wait"
