
from collections import defaultdict
import os
from omegaconf import OmegaConf
import json
import seaborn as sns
import pandas as pd
import matplotlib


_aggregated_data = defaultdict(list)

skip_tasks = ["regression"]
# 1. make the line colors & styles consistent across plots (e.g. the line color for Random differs between Figures 2-5, other line colors & styles vary across plots too). Also, the yellow and orange lines in the current palette are pretty hard to tell apart.
# 2. clean up figure 4 a bit, e.g. larger font size like the other plots, a title more appropriate than "experiment = bios"
task_type = "bad" #options bad or good
graph = "best"

if task_type=="good":
    mydir = "./found_runs_good/"
else:
    mydir = "./found_runs/"

if graph == "best":
    models = [("pretrained" , {}, 'BERT'), 
            ('random', {}, 'Random'),
            ('loaded', {'l_bad_adapted' : '0.0',
                        'l_linear_mi' : '0.0',
                        'l_bad_adapted_grad' : '0.0',
                        'max_adapt_steps' : '0',
                        'train_steps' : '5000'}, 'BERT (tuned, prof.)'),
            ('loaded', {'l_bad_adapted' : '0.0',
                        'l_linear_mi' : '1.0',
                        'l_bad_adapted_grad' : '0.0',
                        'max_adapt_steps' : '16'}, 'MLAC'),
            ('loaded', {'l_bad_adapted' : '1.0',
                'l_linear_mi' : '0.0',
                'l_bad_adapted_grad' : '0.0',
                'max_adapt_steps' : '0'}, 'AC')
                        ]
elif graph == "steps":
    models = [("pretrained" , {}, 'BERT'), 
            ('random', {}, 'Random'),
            ('loaded', {
                        'l_bad_adapted_grad' : '0.0',
                        'max_adapt_steps' : '16'}, '16 steps'),
            ('loaded', {
                'l_bad_adapted_grad' : '0.0',
                'max_adapt_steps' : '4'}, '4 steps'),
            ('loaded', {
                'l_bad_adapted_grad' : '0.0',
                'max_adapt_steps' : '0'}, '0 steps')
                        ]
elif graph == "grad":
    models = [("pretrained" , {}, 'BERT'), 
            ('random', {}, 'Random'),
            ('loaded', {'l_bad_adapted_grad' : '0.0', 'max_adapt_steps' : '16'}, 'No Grad Penalty'),
            ('loaded', {
                'l_bad_adapted_grad' : '1.0', 'max_adapt_steps' : '16'}, 'Grad Penalty')
                        ]
elif graph == "mi":
    models = [("pretrained" , {}, 'BERT'), 
        ('random', {}, 'Random'),
        ('loaded', {'l_linear_mi' : '0.0', 'l_bad_adapted_grad' : '0.0'}, 'No head adjustment'),
        ('loaded', {
            'l_linear_mi' : '1.0', 'l_bad_adapted_grad' : '0.0'}, 'Head adjustment')
                    ]

for dirpath, dirnames, filenames in os.walk(mydir):
    # import pdb; pdb.set_trace()
    if "eval_info.json" in filenames:
        # We're in a result dir
        overrides = OmegaConf.load(os.path.join(dirpath, ".hydra/overrides.yaml"))
        _override_dict = {}
        for override in overrides:
            k, v = override.split("=")
            _override_dict[k] = v

        if _override_dict["experiment"] in skip_tasks:
            continue
        eval_type = _override_dict["eval_network_type"]

        if eval_type not in [x[0] for x in models]:
            continue

        if eval_type == "loaded":
            overrides = OmegaConf.load(os.path.join(dirpath, "loaded_model_conf.yaml"))
            _loaded_model_override_dict = {}
            for override in overrides:
                k, v = override.split("=")
                _loaded_model_override_dict[k] = v
            skip_model = True
            for t, v, name in models:
                if t != "loaded":
                    continue
                all_true = True
                for key, value in v.items():
                    if key not in _loaded_model_override_dict or _loaded_model_override_dict[key] != value:
                        all_true = False
                if all_true:
                    model_name = name
                    skip_model = False
            if skip_model:
                continue
        else:
            model_name = models[[x[0] for x in models].index(eval_type)][2]

        _aggregated_data["Dataset Size"].append(int(_override_dict["adversary.n_examples"]))
        _aggregated_data["experiment"].append(_override_dict["experiment"])
        _aggregated_data["seed"].append(_override_dict["seed"])
        _aggregated_data["Model"].append(model_name)
        with open(os.path.join(dirpath, "eval_info.json")) as f:
            _results = json.load(f)
        if f"eval_only_{task_type}" in _override_dict and _override_dict[f"eval_only_{task_type}"] == "True":
            acc_key = "acc" # TODO: later fix so we append the bad modified again
        else:
            acc_key = f"acc_eval_{task_type}"
        if False:
            if _override_dict["experiment"] == "regression":
                solvesteps_key = "ybad_test_solve_datapoints/0.7"
            else:
                solvesteps_key = "genders_test_solve_datapoints/0.7"
            try:
                _aggregated_data["solve_datapoints"].append(_results[solvesteps_key])
            except:
                # import pdb; pdb.set_trace()
                _aggregated_data["solve_datapoints"].append(_results[solvesteps_key + f"_eval_{task_type}"])
        _aggregated_data["Professions Accuracy (Post-adaptation)"].append(_results[acc_key])

df = pd.DataFrame.from_dict(_aggregated_data)


cmap = reversed(sns.color_palette("colorblind", len(df["Model"].unique()) +5))
print([x for x in cmap])
palette = {
    'AC': (0.8, 0.47058823529411764, 0.7372549019607844),
    'BERT': (0.8352941176470589, 0.3686274509803922, 0.0),
    'BERT (tuned, prof.)': (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
    'MLAC': (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
    'Random' : (0.8705882352941177, 0.677843137254902, 0.0196078431372549),
    '0 steps' : (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
    '16 steps': (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
    '4 steps': "#984ea3",
    "No head adjustment" : (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
    "Head adjustment" : (0.00392156862745098, 0.45098039215686275, 0.6980392156862745)
}
# sns.set(weight="light")
from matplotlib import pyplot
# sns.set_style('darkgrid', {'legend.frameon':True})
import numpy as np
if graph == "best":
    if task_type == "good":
        hue_order = ["Random", "BERT", 'MLAC']
    else:
        hue_order = ["Random", "BERT", 'BERT (tuned, prof.)', "AC", 'MLAC']
elif graph == "steps":
    hue_order = ['Random', 'BERT', '0 steps', '4 steps', '16 steps']
elif graph == "mi":
    hue_order = ['Random', 'BERT', "No head adjustment", "Head adjustment"]


a4_dims = (11.7, 8.27)
# df = mylib.load_data()
fig, ax = pyplot.subplots(figsize=a4_dims)
figure2 = sns.relplot(ax=ax,
    data=df, x="Dataset Size", y="Professions Accuracy (Post-adaptation)", col="experiment",
    hue="Model", style="Model", kind="line", palette=palette, aspect=1.5, hue_order = hue_order
)
ax = figure2.axes[0][0]
ax.set_xscale('log')
pyplot.yticks(fontsize=18)
ax.set_xticks([20,50,100,200], fontsize=18)
ax.set_xticklabels([20,50,100,200], fontsize=18)
ax.set_title(None)

if task_type == "bad":
    ax.set_ylabel("Post-adaptation gender accuracy", fontsize=20)
    ax.set_xlabel("Dataset size", fontsize=20)
else:
    ax.set_ylabel("Post-adaptation professions accuracy", fontsize=20)
    ax.set_xlabel("Dataset size", fontsize=20)



from matplotlib import rc
# rc('font',**{'family':'sans-serif','serif':['Helvetica Neue'], 'weight' : 'light', 'size' : 18})
# # rc('axes',**{'family':'sans-serif','serif':['Helvetica Neue'], 'weight' : 'light', 'size' : 18})
# rc('xtick',**{'family':'sans-serif','serif':['Helvetica Neue'], 'weight' : 'light', 'size' : 18})
# rc('ytick', size=18)
# rc('xlabel',**{'family':'sans-serif','serif':['Helvetica Neue'], 'weight' : 'light', 'size' : 18})
# rc('ylabel',**{'family':'sans-serif','serif':['Helvetica Neue'], 'weight' : 'light', 'size' : 18})
# SMALL_SIZE = 14
MEDIUM_SIZE = 18
# BIGGER_SIZE = 18
# import matplotlib as mpl
# mpl.rcParams['font.family'] = 'Arial'
# mpl.rcParams['font.size'] = 18

hfont = {}
pyplot.rc('font', size=MEDIUM_SIZE, **hfont)          # controls default text sizes
# pyplot.rc('axes', titlesize=MEDIUM_SIZE, **hfont)     # fontsize of the axes title
# pyplot.rc('axes', labelsize=MEDIUM_SIZE, **hfont)    # fontsize of the x and y labels
# pyplot.rc('xtick', labelsize=MEDIUM_SIZE, **hfont)    # fontsize of the tick labels
# pyplot.rc('ytick', labelsize=MEDIUM_SIZE, **hfont)    # fontsize of the tick labels
# pyplot.rc('legend', fontsize=MEDIUM_SIZE, **hfont)    # legend fontsize
# pyplot.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
pyplot.minorticks_off()
# figure2.legend.get_frame().set_facecolor('white')
pyplot.grid(alpha=.5)  #just add this
# pyplot.locator_params(axis='x', nbins=5)
# pyplot.locator_params(axis='y', nbins=8)
if task_type == "good":
    sns.move_legend(figure2, "lower right", bbox_to_anchor=(0.8, .2), frameon=True, facecolor="white", framealpha=1.0, title=None)
else:
    sns.move_legend(figure2, "upper left", bbox_to_anchor=(0.1, .9), frameon=True, facecolor="white", framealpha=1.0, title=None)

# pyplot.legend(frameon=False)
figure2.figure.savefig("out.pdf",bbox_inches='tight')
figure2.figure.savefig("out.png",bbox_inches='tight')

if False:
    sns.relplot(
        data=df, x="Dataset Size", y="solve_datapoints", col="experiment",
        hue="Model", style="Model", kind="line", palette=cmap
    ).figure.savefig("out_datapoints.png")

    # if not dirnames:
    #     print (dirpath, "has 0 subdirectories and", len(filenames), "files")
