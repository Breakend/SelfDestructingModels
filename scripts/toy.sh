if (( "$#" == 0 )); then 
    ID=$(mkid)
else
    ID=$1
fi

echo "Running with id $ID"

ebatch sd_toy_base slconf_jag   "python -m train experiment=regression"
# ebatch sd_toy_newh slconf_jag   "python -m train experiment=regression inner_loop_random_head=True"
# ebatch sd_toy_freeze slconf_jag "python -m train experiment=regression inner_loop_freeze_base=True"

